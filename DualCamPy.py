#!/usr/bin/env python


#
import argparse
import os
import sys
import zwoasi as asi
import cv2
import numpy as np
import matplotlib.pyplot as plt
import datetime
import subprocess
import RPi.GPIO as GPIO
import envirophat as ep
import time
from process_images import *
import threading
import multiprocessing as mpro



env_filename = '/home/pi/asi_drivers/lib/armv7/libASICamera2.so'

parser = argparse.ArgumentParser(description='Process and save images from a camera')
parser.add_argument('filename',
                    nargs='?',
                    help='SDK library filename')
args = parser.parse_args()

# Initialize zwoasi with the name of the SDK library
if args.filename:
    asi.init(args.filename)
elif env_filename:
    asi.init(env_filename)
else:
    print('The filename of the SDK library is required (or set ZWO_ASI_LIB environment variable with the filename)')
    sys.exit(1)



class zwo_controller(object):
    def __init__(self):

        
        num_cameras = asi.get_num_cameras()
        if num_cameras == 0:
            print('No cameras found')
            sys.exit(0)
        
        cameras_found = asi.list_cameras()  # Models names of the connected cameras
        
        if num_cameras == 1:
            self.camera_id = 0
            print('Found one camera: %s' % cameras_found[0])
        else:
            print('Found %d cameras' % num_cameras)
            for n in range(num_cameras):
                print('    %d: %s' % (n, cameras_found[n]))
            # TO DO: allow user to select a camera
            self.camera_id = 0
            print('Using #%d: %s' % (self.camera_id, cameras_found[self.camera_id]))
                  
        self.open_camera(self.camera_id)
        self.set_type(color=False)
    def initial_settings(self):
        self.camera.set_control_value(asi.ASI_BANDWIDTHOVERLOAD,100)
        self.camera.set_control_value(asi.ASI_GAIN,0,auto=False)
        self.camera.set_control_value(asi.ASI_WB_B,90,auto=False)
        self.camera.set_control_value(asi.ASI_WB_R,50,auto=False)
        self.camera.set_control_value(asi.ASI_EXPOSURE,1000,auto=False)
        
        
    def open_camera(self,camera_id):
        self.camera = asi.Camera(camera_id)
        self.camera_info = self.camera.get_camera_property()
        self.controls=self.camera.get_controls()
    def close_camera(self):
        self.camera.close()
        
    def take_dark_image(self):
        image=self.camera.capture()
        return image,self.camera.get_control_values()
    
                
    def take_image(self):
        #self.camera.set_control_value(asi.ASI_BANDWIDTHOVERLOAD,100)
        
        self.camera.start_video_capture()
        
        image=self.camera.capture_video_frame() 
        image=self.camera.capture_video_frame() 
        image=self.camera.capture_video_frame()
       # image=self.camera.capture()

        self.camera.stop_video_capture()
        return image,self.camera.get_control_values()
    def auto_exp(self,starting_exposure=10,auto_wb=False):
        self.check_open()
     
        self.camera.auto_exposure()
        self.camera.set_control_value(asi.ASI_EXPOSURE,starting_exposure,auto=True)
        self.camera.set_control_value(asi.ASI_GAIN,0,auto=True)

        if auto_wb:
            self.camera.auto_wb()
        
        time_in=datetime.datetime.utcnow()+datetime.timedelta(hours=timezone)
        image,output=self.take_image()
        directory_out,filename_out=filename(sky_directory,time_in,'auto')
        file_out=file_checker(directory_out+filename_out,'.jpg')
        if self.type=='raw':
            image = (image/256).astype('uint8')
            image=cv2.cvtColor(image, cv2.COLOR_BAYER_RG2RGB)

        cv2.imwrite(file_out,image)
        save_control_values(directory_out+filename_out,[output])
        self.camera.set_control_value(asi.ASI_EXPOSURE,self.controls['Exposure']['DefaultValue'],auto=False)
        self.camera.set_control_value(asi.ASI_GAIN,self.controls['Gain']['DefaultValue'],auto=False)
        self.camera.set_control_value(asi.ASI_WB_B,self.controls['WB_B']['DefaultValue'],auto=False)
        self.camera.set_control_value(asi.ASI_WB_R,self.controls['WB_R']['DefaultValue'],auto=False)
        to=mpro.Process(target=compute_cloud,args=(image,time_in,configuration_auto))
        to.start()
        exposure=output['Exposure']
        return exposure
        
    def manual_exp(self,exposure,auto_wb=False,gain=0,type_exp='Day'):
        self.check_open()

        self.camera.set_control_value(asi.ASI_GAIN,gain,auto=False)
        self.camera.set_control_value(asi.ASI_WB_B,90,auto=auto_wb)
        self.camera.set_control_value(asi.ASI_WB_R,50,auto=auto_wb)
        self.camera.set_control_value(asi.ASI_EXPOSURE,exposure,auto=False)
        if type_exp=='Day':
            self.camera.set_control_value(asi.ASI_WB_B,90,auto=auto_wb)
            self.camera.set_control_value(asi.ASI_WB_R,50,auto=auto_wb)
            image,output=self.take_image()
        else:
            self.camera.set_control_value(asi.ASI_WB_B,80,auto=auto_wb)
            self.camera.set_control_value(asi.ASI_WB_R,53,auto=auto_wb)
            image,output=self.take_dark_image()
        return image,output
    
    def single_manual(self,exposure,auto_wb=False):
        time_in=datetime.datetime.utcnow()+datetime.timedelta(hours=timezone)

        image,output=self.manual_exp(exposure,auto_wb=auto_wb)
        directory_out,filename_out=filename(sky_directory,time_in,'man')
        file_out=file_checker(directory_out+filename_out,'.jpg')
        if self.type=='raw':
            image = (image/256).astype('uint8')
            image=cv2.cvtColor(image, cv2.COLOR_BAYER_RG2RGB)
        cv2.imwrite(file_out,image)
        
        save_control_values(directory_out+filename_out,[output])
        to=mpro.Process(target=compute_cloud,args=(image,time_in,configuration_auto))
        to.start()

        del image

    def set_type(self,color=False):

        self.check_open()

        if color:
            self.camera.set_image_type(asi.ASI_IMG_RGB24)
            self.type='color'

        else:
            self.camera.set_image_type(asi.ASI_IMG_RAW16)
            self.type='raw'

    def process_hdr(self,image_list,time_in):
        hdr_image=merge_image(image_list)
        directory_out,filename_out=filename(sky_directory,time_in,'hdr')
        file_out=file_checker(directory_out+filename_out,'.jpg')
        cv2.imwrite(file_out,hdr_image)
        print('finished hdr')
        del image_list
        compute_cloud(hdr_image,time_in,configuration_hdr)
        
    def multi_expo(self,exposure_list=[32,64,250,500,1000,1250,1500],gain=0,suffix='Day',auto_wb=False):
        self.check_open()
        image_list=[]
        output_all=[]
        time_in=datetime.datetime.utcnow()+datetime.timedelta(hours=timezone)
        for exposure in exposure_list:
            print('taking exposure '+str(exposure))
            image,output=self.manual_exp(exposure,auto_wb=False,gain=gain,type_exp=suffix)
            image_list.append(image)
            output_all.append(output)
        for i,exposure in enumerate(exposure_list):
            image=image_list[i]
            output=output_all[i]
        
            if self.type=='raw':
                image = (image/256).astype('uint8')
                image=cv2.cvtColor(image, cv2.COLOR_BAYER_RG2RGB)
    
            directory_out=filename2(sky_directory,time_in,suffix)
            filename_out='exposure_'+str(exposure)
            file_out=file_checker(directory_out+filename_out,'.jpg')
            cv2.imwrite(file_out,image)
            save_control_values(directory_out+filename_out,[output])
            image_list[i]=image
        if suffix=='Day':
            #hd=threading.Thread(target=self.process_hdr,args=(image_list,time_in))
            #hd.start()
            hd=mpro.Process(target=self.process_hdr,args=(image_list,time_in))
            hd.start()
            
        else:
            for image in image_list:
                process_dark(image,time_in,configuration_dark)
        del image_list


    def check_open(self):
        if self.camera.closed:
            self.open_camera(self.camera_id)
            
class front_camera_controller(object):
    def auto_exp(self):
        time_in=datetime.datetime.utcnow()+datetime.timedelta(hours=12)
        directory_out,filename_out=filename(front_directory,time_in,'auto')
        file_out=file_checker(directory_out+filename_out,'')

        command_string='fswebcam -d /dev/video0 -r 1920x1080 --set exposure_auto=2 --no-banner '+file_out
        #os.system(command_string)
        prog = subprocess.Popen(command_string.split(), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        returned=prog.communicate()  # Returns (stdoutdata, stderrdata): stdout and stderr are ignored, here
#        if prog.returncode:
#            raise Exception('program returned error code {0}'.format(prog.returncode))

def log_met_data(time_in,heater_status):
    dateout=str('%04d%02d%02d.log' % (time_in.year,time_in.month,time_in.day))
    logfile=dateout
    directory=str(log_directory+'%04d/%02d/%04d%02d%02d' % (time_in.year,time_in.month,time_in.year,time_in.month,time_in.day))
        
    if not os.path.exists(os.path.dirname(directory+"/")):
        try:
            os.makedirs(os.path.dirname(directory+"/"))
        except:
            pass

    if os.path.exists(directory+"/"+logfile)==False:
        l=open(directory+"/"+logfile,'a')
        l.write("Date, Time,SZA, EP_Temperature, EP_Pressure, EP_Heading,Heater_Status\n")
        l.close()
    sza,saz=sunzen_ephem(time_in,latitude,longitude,pressure,temperature)
    l=open(directory+"/"+logfile,'a')

    lineout=("%04d/%02d/%02d,%02d:%02d:%02d,%3.2f,%3.2f,%.2f,%.2f,%d") % (time_in.year,time_in.month,time_in.day,int(time_in.hour), int(time_in.minute), int(time_in.second),sza,ep.weather.temperature(),ep.weather.pressure(),ep.motion.heading(),int(heater_status))
    l.write(lineout+"\n")
    l.close()
if __name__=='__main__':   
    sky_directory='/home/pi/allsky_camera/'
    front_directory='/home/pi/front_facing_camera/'
    log_directory='/home/pi/logs/'
    led_pin=22
    heater_pin=17
    longitude=169.684
    latitude=-45.038
    timezone=12
    pressure=0
    temperature=0
    sky_time_hdr=5
    sky_time_auto=1
    front_time=1
    met_time=1
    sza_threshold=100
    night_exposures=[10000000,60000000,90000000,120000000]
    day_exposures=[16,32,64,250,500,1000,1250,1500]
    
    
    allsky_camera=zwo_controller()
    front_camera=front_camera_controller()
    allsky_camera.set_type(color=True)
    GPIO.setup(heater_pin,GPIO.OUT)
    GPIO.setup(led_pin,GPIO.OUT)
    GPIO.output(led_pin,0)
    heater_status=0
    led_status=0
    time_in=datetime.datetime.utcnow()+datetime.timedelta(hours=timezone)
    next_time=time_in+datetime.timedelta(minutes=1)
    next_time=next_time.replace(second=0,microsecond=0)
    print ('Setup complete')
    exposure_out=500
    while True:
        time_in=datetime.datetime.utcnow()+datetime.timedelta(hours=timezone)
        if (next_time-time_in).total_seconds()<0:
            print('Actions at ',next_time,time_in)
            sza,saz=sunzen_ephem(next_time-datetime.timedelta(hours=timezone),latitude,longitude,pressure,temperature)
            minute_in=next_time.minute
             

            if minute_in % sky_time_auto ==0:
                print('allsky auto')                
                if sza<sza_threshold:
                    exposure_out=allsky_camera.auto_exp(starting_exposure=exposure_out,auto_wb=False)
#                 else:
#                     allsky_camera.auto_exp(starting_exposure=1000000,auto_wb=False)
#
#            if minute_in % front_time ==0:
#                print('front auto')
#
#                front_camera.auto_exp()
                        
        
            if ep.weather.temperature() <=50 and heater_status==0:
                GPIO.output(17,1)
                print('heater on')
                heater_status=1
            if ep.weather.temperature()>50 and heater_status==1:
                GPIO.output(17,0)
                heater_status=0
                print('heater off')
                
            if minute_in % met_time ==0:
                print('met data')
                log_met_data(next_time,heater_status)
                
            if minute_in % sky_time_hdr ==0:
                print('multi exposure')
                try:
                    if sza<sza_threshold:
                        allsky_camera.multi_expo(exposure_list=day_exposures,suffix='Day')
                    else: 
                        allsky_camera.multi_expo(exposure_list=night_exposures,suffix='Night',gain=255)
                except:
                    continue

            next_time=next_time+datetime.timedelta(minutes=1)
            time.sleep(2)
