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
    #images=[cv2.imread(image) for image in images]
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
    
def prepare_mask(image,intensity,sky_indy,spa,psza,configuration):
    
    nanmask=np.ones(image[:,:,0].shape)
    nanmask[np.isnan(sky_indy)]=0

    window=configuration['window']
    if configuration['min_spa']>configuration['window']:
        window+=configuration['min_spa']
    if configuration['sza']>configuration['horizon_limit']:
        window+=configuration['sza']-configuration['horizon_limit']
    sat_count=len([(np.where((intensity>configuration['intensity_limit']) &  (spa<np.deg2rad(window))))][0][0])

    total_count=len([(np.where((spa<np.deg2rad(window))))][0][0])
    
    solar_mask=get_sun_mask(spa,2)


    flag=0.0
    if total_count==0:
        sat_stat=-1.0
        flag+=0.3
    else:
        sat_stat=float(sat_count)/float(total_count)
        
        

        
    if sat_stat>0.1:
        solar_mask=get_sun_mask(spa,configuration['sun_mask_high'])
    elif 0<sat_stat<0.1:
        solar_mask=get_sun_mask(spa,configuration['sun_mask_low'])
    else:
        solar_mask=np.ones(nanmask.shape)
                   
    if configuration['sza']>configuration['horizon_limit'] and sat_stat>0.0:
        sat_stat=1.0
        solar_mask=get_sun_mask(spa,configuration['sun_mask_horizon'])
    horizon_mask=np.ones(nanmask.shape)
    horizon_mask[np.array(psza)>np.deg2rad(configuration['horizon_limit'])]=0
    mask=np.ones(horizon_mask.shape)
    mask[np.any((horizon_mask==0,solar_mask==0,nanmask==0),axis=0)]=0
    
    return mask,sat_stat

def elifan(rbr,mask,bin_vals,sat_stat):

    
    rbr_filtered=rbr[np.where(mask==1)]
    rbr_filtered=rbr_filtered[~np.isnan(rbr_filtered)]
    rbr_filtered=rbr_filtered[np.isfinite(rbr_filtered)]
    cloud_map_elifan=np.zeros(rbr.shape)
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

def process_dark(image,meas_date,configuration):
    directory_out,filename_out=filename(configuration['dark_directory'],meas_date,'corrected')
    file_out=file_checker(directory_out+filename_out,'.jpg')
    #if configuration['dark_image']!=None:
     #   dark_image=cv2.imread(configuration['dark_image'])
     #   image=image-dark_image
    #image=simplest_cb(image,configuration['dark_colour_balance'])
    cv2.imwrite(file_out,image)

def invert_image(image):
    image_out=image.copy()
    image_out[:,:,0]=image[:,:,2]
    image_out[:,:,2]=image[:,:,0]
    return (image_out)

def filename(data_directory,time_in,suffix):
    todayspath=datetime.datetime.strftime(time_in,'%Y/%m/%Y%m%d')
    directory_out=data_directory+todayspath+'/'
    if not os.path.exists(os.path.dirname(directory_out)):
        os.makedirs(os.path.dirname(directory_out))
    n=''
    filename_out=datetime.datetime.strftime(time_in,'%Y%m%d_%H%M_'+suffix)

    return directory_out,filename_out

def filename_day(data_directory,time_in,suffix):
    todayspath=datetime.datetime.strftime(time_in,'%Y/%m/%Y%m%d')
    directory_out=data_directory+todayspath+'/'
    if not os.path.exists(os.path.dirname(directory_out)):
        os.makedirs(os.path.dirname(directory_out))
    n=''
    filename_out=datetime.datetime.strftime(time_in,'%Y%m%d_'+suffix)

    return directory_out,filename_out
def filename2(data_directory,time_in,suffix):
    todayspath=datetime.datetime.strftime(time_in,'%Y/%m/%Y%m%d/%Y%m%d_%H%M')
    directory_out=data_directory+todayspath+'_'+suffix+'/'
    if not os.path.exists(os.path.dirname(directory_out)):
        os.makedirs(os.path.dirname(directory_out))
    return directory_out

def file_checker(filename,extention):
    n=1
    if os.path.exists(filename+extention):
        filename_out=filename+'_'+str(n)+extention
        
        while n<100:
            if os.path.exists(filename_out):
                n+=1
                filename_out=filename+'_'+str(n)+extention
            else:
                break
    else:
        filename_out=filename+extention
    return filename_out
        
def save_control_values(filename, all_settings):
    filename=file_checker(filename,'.txt')
    with open(filename, 'w') as f:
        for settings in all_settings:
            for k in sorted(settings.keys()):
                f.write('%s: %s\n' % (k, str(settings[k])))
    print('Camera settings saved to %s' % filename)
    
def decdeg2dms(dd):
    is_positive = dd >= 0
    dd = abs(dd)
    minutes,seconds = divmod(dd*3600,60)
    degrees,minutes = divmod(minutes,60)
    degrees = degrees if is_positive else -degrees
    return str(int(degrees))+":"+str(int(minutes))+":"+str(seconds)

def sunzen_ephem(time,Lat,Lon,psurf,temp):
    time=time
    observer = ephem.Observer()
    observer.lon = decdeg2dms(Lon)
    observer.lat = decdeg2dms(Lat)
    observer.date = time
 
    observer.pressure=psurf
    observer.temp=temp
    sun = ephem.Sun(observer)
   # sun.compute(observer)
    alt_atr = float(sun.alt)
    solar_altitude=180.0*alt_atr/np.pi
    solar_zenith=90.0-solar_altitude
    solar_azimuth=180*float(sun.az)/np.pi
    return solar_zenith, solar_azimuth

def compute_cloud(image,meas_date,configuration):

    pazi=calculate_azi_zwo2(x0=configuration['x_centre'],y0=configuration['y_centre'],heading=configuration['heading_correction'])
    psza=calculate_pza_zwo(x0=configuration['x_centre'],y0=configuration['y_centre'],proj_grad=configuration['proj_grad'],proj_offset=configuration['proj_offset'])
    sza,azi=sunzen_ephem(meas_date-datetime.timedelta(hours=configuration['timezone']),configuration['latitude'],configuration['longitude'],configuration['pressure'],configuration['temperature'])

    b,g,r=cv2.split(image)
    r=np.array(r,dtype=np.float)
    g=np.array(g,dtype=np.float)
    b=np.array(b,dtype=np.float)
    sky_indy=(b-r)/(b+r)
    intensity=r+b
    rbr=r/b
    if len(rbr[np.isfinite(rbr)])!=0:
        spa=calculate_spa(pazi,psza,azi,sza)
        haze=((b+r)/2.-g)/((b+r)/2.+g)
  
        min_spa=np.nanmin(spa)*180/np.pi
        configuration['min_spa']=min_spa
        configuration['sza']=sza
        flag=0
        mask,sat_stat=prepare_mask(image,intensity,sky_indy,spa,psza,configuration)

        cloud_map_elifan=elifan(rbr,mask,configuration['bin_values'],sat_stat)
        cloud_count_weight=float(sum(psza[cloud_map_elifan==1]))
        clear_count_weight=float(sum(psza[cloud_map_elifan==3]))
        grey_count_weight=float(sum(psza[cloud_map_elifan==2]))
        cloud_map_ids=['Masked','Cloud','Grey/Thin/Unknown','Clear']
        cmap = colors.ListedColormap(['black','white','grey','blue'])
        bounds=[0,1,2,3,4]
        norm = colors.BoundaryNorm(bounds, cmap.N)
        image=cmap(norm(cloud_map_elifan))
        directory_out,filename_out=filename(configuration['cloud_fraction_directory'],meas_date,configuration['image_type']+'_elicm')
        file_out=file_checker(directory_out+filename_out,'.jpg')
        plt.imsave(file_out,image)
        directory_out,filename_out=filename_day(configuration['cloud_fraction_directory'],meas_date,configuration['image_type']+'_elifan')
        filename_out=directory_out+filename_out+'.dat'
        if not os.path.exists(filename_out):
            f=open(filename_out,'w')
            f.write("Daily Log\n Date, Time, Image, Clear Pixels, Cloud Pixels, Grey/Unknown Pixels,Saturation Stat\n")
            f.close()

        f=open(filename_out,'a')
        f.write(str(meas_date)+" "+file_out+" "+"%7d %7d %7d %.3f\n" % (clear_count_weight,cloud_count_weight,grey_count_weight,sat_stat))
        f.close()

configuration_dark={'dark_colour_balance' : 1,
                    'dark_image' : '/home/pi/CamPy4Pi/dark_image.jpg',
                    'image_type' : 'dark',
                    'dark_directory' : '/home/pi/dark_sky/'}
            

configuration_auto={'cloud_fraction_directory' : '/home/pi/cloud_fraction/',
            'x_centre' : 1470,
            'y_centre' : 1165,
            'heading_correction' : 180-21,
            'proj_grad' : 0.057,
            'proj_offset' : 0,
            'sun_mask_high' : 20,
            'sun_mask_low': 10,
            'sun_mask_horizon' : 40,
            'horizon_limit' : 70.,
            'intensity_limit' : 500,
            'window' : 5 ,
            'latitude' : -45.0536,
            'longitude' : 169.6735,
            'pressure' : 1000,
            'temperature' : 10,
            'timezone' : 12,
            'sza_processing_limit' : 90,
            'image_type' : 'auto',
            'bin_values' : [-np.inf,0.35,0.4,0.6,0.7,0.8,np.inf]}
    

configuration_hdr={'cloud_fraction_directory' : '/home/pi/cloud_fraction/',
            'x_centre' : 1470,
            'y_centre' : 1165,
            'heading_correction' : 180-21,
            'proj_grad' : 0.057,
            'proj_offset' : 0,
            'sun_mask_high' : 20,
            'sun_mask_low': 10,
            'sun_mask_horizon' : 40,
            'horizon_limit' : 70.,
            'intensity_limit' : 500,
            'window' : 5 ,
            'latitude' : -45.0536,
            'longitude' : 169.6735,
            'pressure' : 1000,
            'temperature' : 10,
            'timezone' : 12,
            'sza_processing_limit' : 90,
            'image_type' : 'hdr',
            'bin_values' : [-np.inf, 0.3, 0.4, 0.45, 0.65, 0.7,np.inf]}

