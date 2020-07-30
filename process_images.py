# -*- coding: utf-8 -*-
"""
Created on Tue May 15 11:17:06 2018

@author: geddesag

Script that grabs the images from the camera using rsync, creates list of the acquired images,
processes them

"""
#import matplotlib as mpl
#mpl.use('Agg')
#import matplotlib as mpl
#mpl.use('Agg')
import subprocess

import numpy as np

import matplotlib.pyplot as plt
from PIL import Image
from matplotlib import colors    
import os
import cv2
import glob
import ephem
import datetime
import getpass
import shutil
import sys
from solar_calc import *
# from library_dev3 import *



def read_image_raw(file_name):
    imageRaw = cv2.imread(file_name, cv2.IMREAD_GRAYSCALE | cv2.IMREAD_ANYDEPTH)
    rgb = cv2.cvtColor(imageRaw, cv2.COLOR_BAYER_RG2RGB)
    return rgb

def weka_dir(dir_in,on_weka=False):
    
    username=getpass.getuser()
    if 'win' in sys.platform:
        on_weka=False
    else:
        on_weka=True
    if on_weka:
        dir_out=dir_in.replace('\\','/')
        if 'Q:' in dir_out:
            dir_out=dir_out.replace('Q:','/mnt/'+username+'/groups')
        if 'V:' in dir_out:
            dir_out=dir_out.replace('V:','/mnt/'+username+'/groups/atmos')
        if 'S:' in dir_out:
            dir_out=dir_out.replace('S:','/mnt/geddesag/scratch')

        # if '/weka/' in dir_out:
        #     dir_out=dir_out.replace('/weka/','/')
        if '//weka/uvuser/' in dir_out:
            #print ('helloooo')
            dir_out=dir_out.replace('//weka/uvuser/','/home/uvuser/')
            
    else:
        dir_out=dir_in
    return dir_out    

def merge_image(images):
    images=[cv2.imread(image) for image in images]
    alignmtb=cv2.createAlignMTB()
    alignmtb.process(images,images)
    mm=cv2.createMergeMertens()
    result_mertens=mm.process(images)
    res_mertens_8bit = np.clip(result_mertens*255, 0, 255).astype('uint8')
    img_out=res_mertens_8bit
    

    return img_out



def preprocess_image(image,cb=10,gamma=2.):
    if cb>0:
        image=simplest_cb(image,cb)
    if gamma!=0:
        image=adjust_gamma(image,gamma=gamma)
    return image

def apply_mask(matrix, mask, fill_value):
    masked = np.ma.array(matrix, mask=mask, fill_value=fill_value)
    return masked.filled()

def apply_threshold(matrix, low_value, high_value):
    low_mask = matrix < low_value
    matrix = apply_mask(matrix, low_mask, low_value)

    high_mask = matrix > high_value
    matrix = apply_mask(matrix, high_mask, high_value)

    return matrix

def simplest_cb(img, percent):
    assert img.shape[2] == 3
    assert percent > 0 and percent < 100

    half_percent = percent / 200.0

    channels = cv2.split(img)

    out_channels = []
    for channel in channels:
        assert len(channel.shape) == 2
        # find the low and high precentile values (based on the input percentile)
        height, width = channel.shape
        vec_size = width * height
        flat = channel.reshape(vec_size)
        flat=flat[flat>=0]
        assert len(flat.shape) == 1

        flat = np.sort(flat)

        n_cols = flat.shape[0]
        low_val  = flat[int(np.math.floor(int(n_cols) * half_percent))]

        high_val = flat[int(np.math.ceil( n_cols * (1.0 - half_percent)))]

#        print "Lowval: ", low_val
#        print "Highval: ", high_val

        # saturate below the low percentile and above the high percentile
        thresholded = apply_threshold(channel, low_val, high_val)
        # scale the channel
        normalized = cv2.normalize(thresholded, thresholded.copy(), 0, 255, cv2.NORM_MINMAX)
        out_channels.append(normalized)

    return cv2.merge(out_channels)    

def date_str_to_date(x,y,z):

    out=datetime.datetime.strptime(x+y,"%Y/%m/%d%H:%M:%S")
    return out
    
def adjust_gamma(image, gamma=1.0):
	# build a lookup table mapping the pixel values [0, 255] to
	# their adjusted gamma values
	invGamma = 1.0 / gamma
	table = np.array([((i / 255.0) ** invGamma) * 255
		for i in np.arange(0, 256)]).astype("uint8")
 
	# apply gamma correction using the lookup table
	return cv2.LUT(image, table) 
def invert_image(image):
    image_out=image.copy()
    image_out[:,:,0]=image[:,:,2]
    image_out[:,:,2]=image[:,:,0]
    return (image_out)
def display_image(image,gamma=2.2,cb=0):
    image=preprocess_image(image,gamma=gamma,cb=cb)
    
    plt.figure()
    plt.imshow(invert_image(image))
    b,g,r=cv2.split(image)
    
    r=np.array(r,dtype=np.float)
    g=np.array(g,dtype=np.float)
    b=np.array(b,dtype=np.float)
    sky_indy=(b-r)/(b+r)
    plt.figure()
    plt.imshow(sky_indy,vmin=-0.5,vmax=0.5)
    
def prepare_mask(image,intensity,horizon_limit,window=5,sun_mask_high=20,sun_mask_low=10,sun_mask_horizon=40,intensity_limit=500):
    
    nanmask=np.ones(image[:,:,0].shape)
    nanmask[np.isnan(sky_indy)]=0


    if min_spa>window:
        window+=min_spa
    if sza>horizon_limit:
        window+=sza-horizon_limit
    sat_count=len([(np.where((intensity>intensity_limit) &  (spa<np.deg2rad(window))))][0][0])

    total_count=len([(np.where((spa<np.deg2rad(window))))][0][0])
    
    solar_mask=get_sun_mask(spa,2)


    flag=0.0
    if total_count==0:
        sat_stat=-1.0
        flag+=0.3
    else:
        sat_stat=float(sat_count)/float(total_count)
        
        

        
    if sat_stat>0.1:
        solar_mask=get_sun_mask(spa,sun_mask_high)
    elif 0<sat_stat<0.1:
        solar_mask=get_sun_mask(spa,sun_mask_low)
    else:
        solar_mask=np.ones(np.array(r).shape)
                   
    if sza>horizon_limit and sat_stat>0.0:
        sat_stat=1.0
        solar_mask=get_sun_mask(spa,sun_mask_horizon)
    horizon_mask=np.ones(np.array(r).shape)
    horizon_mask[np.array(psza)>np.deg2rad(horizon_limit)]=0
    haze_mask=np.ones(horizon_mask.shape)
    mask=np.ones(r.shape)
    mask[np.any((horizon_mask==0,solar_mask==0,haze_mask==0,nanmask==0),axis=0)]=0
    
    return mask,sat_stat

def elifan(rbr,mask,bin_vals,sat_stat):

    
    rbr_filtered=rbr[np.where(mask==1)]
    rbr_filtered=rbr_filtered[~np.isnan(rbr_filtered)]
    rbr_filtered=rbr_filtered[np.isfinite(rbr_filtered)]
    cloud_map_elifan=np.zeros(r.shape)
    cloud_map_elifan[mask==0]=0
    if len(rbr_filtered)!=0:
        counts,bins=np.histogram(rbr_filtered,bins=bin_vals)
        counts_fine,bins_fine=np.histogram(rbr_filtered,bins=np.arange(0,1.01,0.01))
        total_counts=np.sum(counts)
        pdf=counts_fine/total_counts
        
        cloud_frac_1=1-np.sum(counts[:3])/total_counts
        sum_mid=np.sum(counts[1:3])/total_counts
        sum_low=np.sum(counts[0:2])/total_counts
        max_mid=max(pdf[int(bin_vals[1]/0.01):int(bin_vals[3]/0.01)])
        
        min_val=bin_vals
    
        cloud=1
        if cloud_frac_1>0.98 or np.nanmax(rbr_filtered)<bin_vals[3]:
            print('all_cloud')
            cloud_map_elifan[mask==1]=1
            cloud=0
        elif sat_stat==0.0 and sum_low<0.125:
            print('all_cloud')
            cloud_map_elifan[mask==1]=1
            cloud=0
        elif cloud_frac_1<0.025 and counts[-1]!=0:
            print('all_clear')
            cloud_map_elifan[mask==1]=3
    
            tests=[max_mid>0.35,sum_mid>0.90,sum_low<0.125]
            if sum(tests)>=2:
                cloud=1
        if cloud==1:
            print('partly')
    
    
            cloud_map_elifan[np.logical_and(mask==1,rbr<bin_vals[3])]=3
            cloud_map_elifan[np.logical_and(mask==1,rbr>bin_vals[3],rbr<bin_vals[4])]=2
            cloud_map_elifan[np.logical_and(mask==1,rbr>bin_vals[4])]=1

       

    return cloud_map_elifan

def calculate_azi_zwo2(image_pixel_radius=1040,x0=1548,y0=1040,heading=0):
    """calculates the azmimuth of each pixel
    
    image_pixel_radius: radius of the circular part of the image, assuming it is is centred
    
    """
    
    
    azimuth=np.zeros((2080, 3096))

    x=np.resize(np.arange(0,len(azimuth[0])),(2080,3096))
    y=np.resize(np.arange(0,len(azimuth)),(3096,2080)).T
    x=x-float(x0)
    y=y-float(y0)
    theta_im=np.zeros((2080, 3096))
    theta_im=np.arctan(y/x)
    azimuth[np.where((x>=0) & (y>0))]=np.deg2rad(90.)-(np.deg2rad(heading)+theta_im[np.where((x>=0) & (y>0))])
    azimuth[np.where((x<0) & (y>=0))]=np.pi+np.deg2rad(90.)-(np.deg2rad(heading)+theta_im[np.where((x<0) & (y>=0))])
    azimuth[np.where((x<=0) & (y<0))]=np.pi+np.deg2rad(90.)-(np.deg2rad(heading)+theta_im[np.where((x<=0) & (y<0))])
    azimuth[np.where((x>0) & (y<=0))]=np.deg2rad(90.)-(np.deg2rad(heading)+theta_im[np.where((x>0) & (y<=0))])
    azimuth[np.where((x>=0) & (y>=0))]=np.deg2rad(90.)-(np.deg2rad(heading)+theta_im[np.where((x>=0) & (y>=0))])



    azimuth[azimuth<0]=2*np.pi+azimuth[azimuth<0]
    azimuth[azimuth>2*np.pi]=azimuth[azimuth>2*np.pi]-2*np.pi
    

    return azimuth

def calculate_pza_zwo(x0=1548,y0=1040,proj_grad=0.06,proj_offset=0):
  
    zenith=np.zeros((2080, 3096))

    x=np.resize(np.arange(0,len(zenith[0])),(2080,3096))
    y=np.resize(np.arange(0,len(zenith)),(3096,2080)).T
    
    
    zenith=np.deg2rad(((((y-y0)**2+(x-x0)**2))**0.5*proj_grad+proj_offset))

    return zenith

def calculate_spa(pixel_azimuth,pixel_zenith,sun_azimuth,sun_zenith):
    A=np.deg2rad(sun_zenith)
    C=pixel_zenith
    b=((np.deg2rad(sun_azimuth)-pixel_azimuth)**2)**0.5
    
    B=np.arccos(np.cos(A)*np.cos(C)+np.sin(A)*np.sin(C)*np.cos(b))
    return B

def get_sun_mask(spa,angle):
    sun_mask=np.ones(spa.shape)
    sun_mask[spa<np.deg2rad(angle)]=0
    return sun_mask
    
on_weka=True
   
pi_number_in=3
pi_number="%02d" % pi_number_in
lat=-45.0536
lon=169.6735
#sys.stdout= open("/home/uvuser/pi03_python_log3.dat","a")
#sys.stderr=sys.stdout
start_date=datetime.datetime(2020,1,1)
end_date=datetime.datetime(2020,7,1)
image_type='Day'
run_date=start_date
delete=False
x_centre=1470
y_centre=1165
pazi=calculate_azi_zwo2(x0=x_centre,y0=y_centre,heading=180-21)
psza=calculate_pza_zwo(x0=x_centre,y0=y_centre,proj_grad=0.057,proj_offset=0)
while run_date<=end_date:
    """ Lets do this nicely"""
    
    directory_to_search=weka_dir(datetime.datetime.strftime(run_date,'V:/uv/allsky/pi03/%Y/%m/%Y%m%d/'))
    if image_type=='auto':
        image_list=glob.glob(directory_to_search+'*'+image_type+'.jpg')
    else:
        image_list=glob.glob(directory_to_search+'*'+image_type)
    
    for item in image_list:
        if image_type=='auto':
            image=cv2.imread(item)
            bin_values=[-np.inf,0.35,0.4,0.6,0.7,0.8,np.inf]
        if image_type=='Day':
            if os.path.exists(item+'.jpg'):
                image=cv2.imread(item+'.jpg')
            else:
                images=glob.glob(item+'/*.jpg')
                image=merge_image(images)
                if delete==True:
                    shutil.rmtree(item)
                cv2.imwrite(item+'.jpg',image)
            bin_values=[-np.inf, 0.4, 0.45, 0.6, 0.65, 0.7,np.inf]
        if image_type=='Night':
            print(item)
            images=glob.glob(item+'/*.jpg')
            for image_in in images:
                image=cv2.imread(image_in)
                image=simplest_cb(image,1)
                name_out=image_in.split('.')[0]
                cv2.imwrite(name_out+'_corrected.jpg',image)
        if image_type!='Night':
            if on_weka==False:
                time_info=item.split('\\')[1].split('_')
            else:
                time_info=item.split('/')[-1].split('_')

            meas_date=datetime.datetime.strptime(time_info[0]+time_info[1],'%Y%m%d%H%M')
            
            print(meas_date)
            sza,azi=sunzen_ephem(meas_date-datetime.timedelta(hours=12),-45.038,169.684,1000,10)
            if sza<=90:
                # image_hsv=cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
                # h,s,v=cv2.split(image)
                # v=cv2.equalizeHist(v)
                # image_hsv_corrected=cv2.merge([h,s,v])
                # image=cv2.cvtColor(image_hsv_corrected,cv2.COLOR_HSV2BGR)
                b,g,r=cv2.split(image)
               # h,s,v=cv2.split(image_hsv)

                r=np.array(r,dtype=np.float)
                g=np.array(g,dtype=np.float)
                b=np.array(b,dtype=np.float)
                sky_indy=(b-r)/(b+r)
                haze_hannover=((b+r)/2.-g)/((b+r)/2.+g)
                intensity=r+b
                rbr=r/b
                if len(rbr[np.isfinite(rbr)])!=0:
                    #-90+6

                   # adjustment=calculate_pza_slope_adjust(pazi2,120+i*10,2*k)

                    spa=calculate_spa(pazi,psza,azi,sza)
                    haze=((b+r)/2.-g)/((b+r)/2.+g)
                    
                    spa_limit=15
                    horizon_limit=70.
                    intensity_limit=500
                    window=5
                    min_spa=np.nanmin(spa)*180/np.pi
                    spa_max=min_spa+120
                    flag=0
                    mask,sat_stat=prepare_mask(image,intensity,horizon_limit,window=window,sun_mask_high=20,sun_mask_low=10,sun_mask_horizon=40,intensity_limit=intensity_limit)
                    
                    
                    cloud_map_elifan=elifan(rbr,mask,bin_values,sat_stat)
                    
                    cloud_count_weight=float(sum(psza[cloud_map_elifan==1]))
                    clear_count_weight=float(sum(psza[cloud_map_elifan==3]))
                    grey_count_weight=float(sum(psza[cloud_map_elifan==2]))
                    
                    cloud_map_ids=['Masked','Cloud','Grey/Thin/Unknown','Clear']
                    cmap = colors.ListedColormap(['black','white','grey','blue'])
                    bounds=[0,1,2,3,4]
                    norm = colors.BoundaryNorm(bounds, cmap.N)
                    image=cmap(norm(cloud_map_elifan))
                    
                    image_name_out=weka_dir(datetime.datetime.strftime(meas_date,'V:/uv/allsky/pi03/%Y/%m/%Y%m%d/%Y%m%d_%H%M_'+image_type+'_elicm_nv.jpg'))
                    plt.imsave(image_name_out,image)
                    
                    filename_out=weka_dir(datetime.datetime.strftime(meas_date,'V:/uv/allsky/pi03/%Y/%m/%Y%m%d/%Y%m%d_'+image_type+'_elifan.dat'))
                    if not os.path.exists(filename_out):
                        f=open(filename_out,'w')
                        f.write("Daily Log\n Date, Time, Image, Clear Pixels, Cloud Pixels, Grey/Unknown Pixels,Saturation Stat\n")
                        f.close()
            
                    x=0
                    while x<5:
                        try:
                            f=open(filename_out,'a')
                            f.write(str(meas_date)+" "+item+" "+"%7d %7d %7d %.3f\n" % (clear_count_weight,cloud_count_weight,grey_count_weight,sat_stat))
                            f.close()
                            x=6
                        except IOError:
                            
                            print("trying again")
                            x=x+1
                            continue
    run_date=run_date+datetime.timedelta(hours=24)
        
#        sky_indy_temp,band1,mean_band2,std_band2,r_squared,mean_corr,flag=correct_sky_index(sky_indy,haze,mask,horizon_limit,spa_limit,intensity_limit,spa_max)
#        
#        
#        
#        hsc,hcc,hgc,cloud_map_niwa,sat_stat,b1_threshold_sky_indy1_low,b1_threshold_sky_indy1_high,flag,modal_flag,r_squared=niwa(sky_indy_temp,mask,band1,mean_band2,std_band2,mean_corr,sat_stat)
#
#        
#        
#        cloud_map_ids=['Masked','Cloud','Grey/Thin','Clear']
#        cmap = colors.ListedColormap(['black','white','grey','blue'])
#        bounds=[0,1,2,3,4]
#        norm = colors.BoundaryNorm(bounds, cmap.N)
#        image=cmap(norm(cloud_map_niwa))
#        
#        plt.imsave(datetime.datetime.strftime(meas_date,'V:/uv/allsky/pi03/%Y/%m/%Y%m%d/%Y%m%d_%H%M_'+image_type+'_cm.jpg'),image)
#        
#    run_date=run_date+datetime.timedelta(hours=24)
    
#    
#[-0.22456711 -0.31945197  0.14638141  0.50532148]
#[-0.19683721 -0.32365105  0.06395923  0.44956627]
#[-0.20968447 -0.28426046  0.10100695  0.45682384]
      #  cv2.imwrite(item.split('.')[0]+'_2.jpg',image)
            

#    picmd=datetime.datetime.strftime(run_date,"find /mnt/uvuser/groups/atmos/uv/allsky/pi03/%Y/%m/%Y%m%d/ -name \*.jpg")
#    pi_return=subprocess.check_output(picmd,shell=True)
#    image_list=[]
#    if ".jpg" in pi_return:
#        print "Images Acquired"
#        pi02_return=pi_return.split("\n")
#        for line in pi02_return:
#            if ".jpg" in line:
#                image_list.append(line)
#                
#    image_groups={}
#    for image_in in image_list:
#        name=image_in.split('/')[-1][:-6]
#        if name not in image_groups:
#            image_groups[name]=[image_in]
#        else:
#            image_groups[name].append(image_in)
#            
#    image_groups.pop('', None)     
#    directory_in="/mnt/uvuser/groups/atmos/uv/allsky/pi"+pi_number+"/"
#    for key in sorted(image_groups.keys()):
#        if len(image_groups[key])==4:
#            print "Night, all images acquired"
#            for image in image_groups[key]:
#                try:
#                    img=read_image_raw(image)
#                    img=simplest_cb(img,1)
#                    cv2.imwrite(image[:-4]+"_processed.jpeg",img)
#                    os.remove(image)
#                except:
#                    print 'bad image'
#                    continue
#    
#        if len(image_groups[key])==8:
#            print key
#            print "Day, all images acquired"
#                
#    
#            try:
#                image_list=[read_image_raw(fn) for fn in image_groups[key]]   
#                image_list_cleaned=[]
#                if len(where(image_list[0][:,:,0]>250))>1000:
#                    print 'saturated pixels'
#                exp_list=[32e-6,64e-6,250e-6,500e-6,1000e-6,2000e-6,5000e-6,10000e-6]
#                exp_cleaned=[]
#                for i,image_in in enumerate(image_list):
#                   # print float(len(where(image_list[i][:,:,0]>250)[0]))
#                   # print float(len(where(image_list[i][:,:,0]>-1.)[0]))
#        #                if np.average(image_in[:,:,2])>np.average(image_in[:,:,0]):
#        #                    print 'bad image, excluding'
#        
#    #                if i>=4:#float(len(np.where(image_list[i][:,:,0]>250)[0]))/float(len(np.where(image_list[i][:,:,0]>-1)[0]))>0.25:
#    #                    print 'too saturated, excluding'
#    #                else:
#                    if i<5:
#                        exp_cleaned.append(exp_list[i])
#                        image_list_cleaned.append(image_in)
#                exp_cleaned=np.array(exp_cleaned,dtype=np.float32)
#                alignmtb=cv2.createAlignMTB()
#                alignmtb.process(image_list_cleaned,image_list_cleaned)
#                mm=cv2.createMergeMertens()
#                result_mertens=mm.process(image_list_cleaned)
#                res_mertens_8bit = clip(result_mertens*255, 0, 255).astype('uint8')
#                img_out=res_mertens_8bit
#                img_out=simplest_cb(img_out,10)
#                img_out=adjust_gamma(img_out,gamma=2.)
#                cv2.imwrite(fn[:-6]+"_merged_all.jpg",img_out) 
#                for fn in image_groups[key]:
#                    os.remove(fn)
#            except:
#                print "bad merge, pass"
#                continue
#            #fn=image_groups[key][0]
##                      fn=image_in
#            date_string=fn.split('/')[-1]
#            date_string=date_string.rstrip('_merged_all.jpg')
#           # date_string+="00"
#          #  date_string='201806241030'
#            print date_string
#            meas_date=datetime.datetime.strptime(date_string,'%Y%m%d%H%M%S')
#            meas_date=meas_date.replace(second=0)
##        print meas_date
##        sza,azi=sunzen_ephem(meas_date-datetime.timedelta(hours=12),-45.038,169.684,1000,10)
##        image_in=directory_in+fn[:-6]+"_merged_all.jpg"
#            print meas_date
#            sza,azi=sunzen_ephem(meas_date-datetime.timedelta(hours=12),-45.038,169.684,1000,10)
#            #directory_in='V:/uv/allsky/pi03/2018/06/20180624/'
#            image_in=fn[:-6]+"_merged_all.jpg"
#            print image_in
#            #image_in=directory_in+date_string+"_merged_all.jpeg"
#            
#    
#            img=cv2.imread(image_in)
#            #    bob=simplest_cb(img,1)
#            b,g,r=cv2.split(img)
#            
#            r=np.array(r,dtype=np.float)
#            g=np.array(g,dtype=np.float)
#            b=np.array(b,dtype=np.float)
#            #-90+6
#            x_centre=1470
#            y_centre=1165
#           # adjustment=calculate_pza_slope_adjust(pazi2,120+i*10,2*k)
#            pazi=calculate_azi_zwo2(r,x0=x_centre,y0=y_centre,heading=180-21)
#            psza=calculate_pza_zwo(r,x0=x_centre,y0=y_centre,proj_grad=0.057,proj_offset=0)
#            
#            spa=calculate_spa(pazi,psza,azi,sza)
#
#            #
#            sky_indy=(b-r)/(b+r)
#            nanmask=np.ones(np.array(r).shape)
#            nanmask[np.isnan(sky_indy)]=0
#            haze_hannover=((b+r)/2.-g)/((b+r)/2.+g)
#            intensity=r+b
#            intensity_limit=500
#            spa_limit=20
#            horizon_limit=75.
#            window=5
#            min_spa=np.nanmin(spa)*180/pi
#
#            if min_spa>window:
#                window+=min_spa
#            if sza>horizon_limit:
#                window+=sza-horizon_limit
#            sat_count=len([(np.where((intensity>intensity_limit) &  (spa<np.deg2rad(window))))][0][0])
#    #        sat_count=len([(where((intensity>intensity_limit)))][0][0])
#    #        total_count=intensity.size
#            total_count=len([(where((spa<np.deg2rad(window))))][0][0])
#            
#            solar_mask=get_sun_mask(spa,2)
#    
#         #   imshow((r+g+b)*solar_mask)
#        
#            flag=0.0
#            if total_count==0:
#                sat_stat=-1.0
#                flag+=0.3
#            else:
#                sat_stat=float(sat_count)/float(total_count)
#                
#                
#            if sat_stat>0.1:
#                sat_stat=1.
#                flag=1.0
#                
#            if sat_stat>0.5:
#                print sat_stat
#                solar_mask=get_sun_mask(spa,20)
#            elif 0<sat_stat<0.1:
#                print sat_stat
#                solar_mask=get_sun_mask(spa,10)
#            else:
#                solar_mask=np.ones(np.array(r).shape)
#                
#                    
#            if sza>horizon_limit and sat_stat>0.0:
#                sat_stat=1.0
#                spa_limit=20
#                solar_mask=get_sun_mask(spa,40)
#            horizon_mask=np.ones(np.array(r).shape)
#            horizon_mask[np.array(psza)>np.deg2rad(horizon_limit)]=0
#            haze_mask=np.ones(horizon_mask.shape)
#            mask=np.ones(r.shape)
#            mask[np.any((horizon_mask==0,solar_mask==0,haze_mask==0,nanmask==0),axis=0)]=0
#            
#            cloud_map_ids=['Masked','Cloud','Grey/Thin','Clear']
#            cmap = colors.ListedColormap(['black','white','grey','blue'])
#            bounds=[0,1,2,3,4]
#            norm = colors.BoundaryNorm(bounds, cmap.N)
#            print r.max(),g.max(),b.max(),sky_indy.max()
#            try:
#                hsc,hcc,hgc,cloud_map_niwa,sat_stat,b1_threshold_sky_indy1_low,b1_threshold_sky_indy1_high,flag,modal_flag,r_squared=niwa(sky_indy,haze_hannover,intensity,psza,pazi,spa,mask,spa_limit,min_spa+120.,intensity_limit,horizon_limit,sat_stat,flag,sza)
#            
#            
#            except:
#                print 'failed, bad image?'
#                continue
#                
#           # imshow(cloud_map_niwa)
#           
#            year=str(meas_date.year)
#            month="%02d" % meas_date.month
#            day="%02d" % meas_date.day
#            suffix=0
#            todayspath=year+month+day
#            
#            cmap=cmap
#            norm=norm
#            image=cmap(norm(cloud_map_niwa))
#            directory='/mnt/uvuser/groups/atmos/uv/allsky/pi03/'+year+'/'+month+'/'+todayspath+'/'
#            date_string=datetime.datetime.strftime(meas_date,'%Y%m%d%H%M')
#            print directory+date_string+"_cmap_niwa_2.jpeg"
#            plt.imsave(directory+date_string+"_cmap_niwa.jpeg",image)
#    
#
#            filepath_out=todayspath
#            
#            
#            #dir_out="/mnt/geddesag/scratch/geddesag/AllskyPi/all/"+folder_in+"/"
#            filename_out=directory+datetime.datetime.strftime(meas_date,'%Y%m%d')+"_merged.log"
#        
#            print filename_out
#            if not os.path.exists(filename_out):
#                f=open(filename_out,'w')
#                f.write("Daily Log\n Date, Time, Image, Clear Pixels, Cloud Pixels, Grey Pixels, Sun Saturation, Threshold 1, Threshold 2, Flag, Modal Flag, R-Squared, mean blue, mean green, mean r\nFlag IDs +1 cloud detected, +0.1 solar correction, +0.3 sun out of image, cannot correct\n Modal Flags 0 fwd3, 1 rvs3, 2 fwd2, 3 rvs2")
#                f.close()
#    
#            x=0
#            while x<5:
#                try:
#                    f=open(filename_out,'a')
#                    f.write(str(meas_date)+" "+image_in+" "+"%7d %7d %7d %.3f %.3f %.3f %.2f %d %.3f %.2f %.2f %.2f\n" % (hsc,hcc,hgc,sat_stat,b1_threshold_sky_indy1_low,b1_threshold_sky_indy1_high,flag,modal_flag,r_squared,b[0][0],g[0][0],r[0][0]))
#                    f.close()
#                    x=6
#                except IOError:
#                    
#                    print "trying again"
#                    x=x+1
#                    continue
#                
#    run_date=run_date+datetime.timedelta(hours=24)
