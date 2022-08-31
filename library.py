# -*- coding: utf-8 -*-
"""
Created on Mon Aug 31 15:49:23 2020

@author: geddesag
"""

import pandas as pd
import base64
import datetime
import requests
import os
import numpy as np
import ephem
import  math




def decdeg2dms(dd):
    """Converts decimal degrees to degrees minutes seconds"""
    is_positive = dd >= 0
    dd = abs(dd)
    minutes,seconds = divmod(dd*3600,60)
    degrees,minutes = divmod(minutes,60)
    degrees = degrees if is_positive else -degrees
    return str(int(degrees))+":"+str(int(minutes))+":"+str(seconds)
      
def sunzen_ephem(time,Lat,Lon,psurf,temp,alt):
    
    observer = ephem.Observer()
    observer.lon = decdeg2dms(Lon)
    observer.lat = decdeg2dms(Lat)
    observer.date = time
    observer.elevation=alt

    observer.pressure=psurf
    observer.temp=temp
    observer.horizon=math.acos(ephem.earth_radius / (alt + ephem.earth_radius))
    sun = ephem.Sun(observer)
   # sun.compute(observer)
    alt_atr = float(sun.alt)
    solar_altitude=180.0*alt_atr/np.pi
    solar_zenith=90.0-solar_altitude
    solar_azimuth=180*float(sun.az)/np.pi
    return solar_zenith, solar_azimuth
    
def sunrise_sunset(date,Lat,Lon,psurf,temp,alt):
    observer = ephem.Observer()
    observer.lon = decdeg2dms(Lon)
    observer.lat = decdeg2dms(Lat)
    observer.date = date
    observer.elevation=alt
    observer.pressure=psurf
    observer.temp=temp
    observer.horizon=math.acos(ephem.earth_radius / (alt + ephem.earth_radius))
    sun = ephem.Sun(observer)
   # sun.compute(observer)
    rising=observer.next_rising(sun,start=observer.date)
    setting=observer.next_setting(sun,start=observer.date)
    return rising.datetime(),setting.datetime()



def wind_direction_as_str(winddirection):
    wind_string='N/A'
    if 360>=winddirection>337.5 or winddirection<=22.5:
        wind_string='N'
    if 22.5<winddirection<=67.5:
        wind_string='NE'
    if 67.5<winddirection<=112.5:
        wind_string='E'
    if 122.5<winddirection<=157.5:
        wind_string='SE'
    if 157.5<winddirection<=202.5:
        wind_string='S'
    if 202.5<winddirection<=247.5:
        wind_string='SW'
    if 247.5<winddirection<=292.5:
        wind_string='W'
    if 292.5<winddirection<=337.5:
        wind_string='NW' 
        
    return wind_string

def get_most_recent_allsky(window):
    exists=False
    time_now=datetime.datetime.utcnow()+datetime.timedelta(hours=12)
    ts=time_now
    search_directory=datetime.datetime.strftime(time_now,'//LAUWFS01/groups$/atmos/uv/allsky/pi05/imagery/%Y/%m/%Y%m%d/')
    try:
        directory_list=smbclient.listdir(search_directory)
        while exists==False:
            filename_to_check=datetime.datetime.strftime(time_now,'%Y%m%d_%H%M_hdr.jpg')
            if filename_to_check in directory_list:
                exists=True
                filename_to_check=search_directory+filename_to_check
            else:
                time_now=time_now-datetime.timedelta(minutes=1)
            if (ts-time_now).total_seconds()>window*60:
                filename_to_check=None
                exists=True
    except Exception as e:
        print(e)
        filename_to_check=None
        
    

    return filename_to_check,time_now

def find_nearest(array,value):
    idx = np.argmin(np.abs(array-value))
    return idx
 
def check_if_samba_file_exists(file_in):
    try:
        smbclient.stat(file_in)
        return True
    except Exception as x:# smbprotocol.exceptions.SMBOSError:
        print(x)
        return False
        
def read_dobson_file(dobson_file):
    with smbclient.open_file(dobson_file, mode="r",share_access='rwd') as fd:
        content=fd.readlines()
   # content=content[::3]
    times=[]
    obs_flag=[]
    ozone=[]
    w_p=[]
    obs_type=[]
    for line in content:
        if line[10]=='0' or line[10]=='1':
            if line[14:20]!='999999':
                obs_flag.append(line[11])
                ozone.append(int(line[61:64]))
                w_p.append(int(line[65]))
                obs_type.append(int(line[66:68]))
                times.append(datetime.datetime.strptime(line[0:8]+line[14:20],"%Y%m%d%H%M%S"))
        
    dobson_data={'time' : np.array(times), 'ozone' : ozone, 'obs_flag' : obs_flag, 'obs_type' : obs_type, 'wav_pair' : w_p}
        

    return dobson_data     
   
def get_toc_direct(meas_datetime,toc_offset=12):
    meas_datetime=meas_datetime+datetime.timedelta(hours=toc_offset)
    
    dobson_file=datetime.datetime.strftime(meas_datetime,"//LAUWFS01/groups$/atmos/ozone/dobson/Data_D072/%Y/Obs_data/ttl%y%m.LDR")
    found_dobson=False
    while found_dobson==False:
      
      dobson_file_t=datetime.datetime.strftime(meas_datetime,"//LAUWFS01/groups$/atmos/ozone/dobson/Data_D072/%Y/Obs_data/ttl%y%m.LDR")
      if check_if_samba_file_exists(dobson_file_t):
        if smbclient.stat(dobson_file_t).st_size>0:
          dobson_file=dobson_file_t
          found_dobson=True
      meas_datetime-=datetime.timedelta(hours=24)
    #print(dobson_file)
    #dic_out={}
    """
    """  
    if check_if_samba_file_exists(dobson_file) and smbclient.stat(dobson_file).st_size>0:
        print(dobson_file)
        data=read_dobson_file(dobson_file)
        
        """
        Find the nearest time to the obstime
        """
        
        dob_index=find_nearest(data['time'],meas_datetime)
        
        """
        check if time>12 hours difference
        """
        ozone_out=float(data['ozone'][dob_index])
        time_out=data['time'][dob_index]
        wp_out=data['wav_pair'][dob_index]
        type_out=data['obs_type'][dob_index]
        dechr_out=float((time_out-datetime.timedelta(hours=toc_offset)).hour+(time_out-datetime.timedelta(hours=toc_offset)).minute/60.)
        
        if np.abs((data['time'][dob_index]-meas_datetime).total_seconds()/3600.)>96:
            
            print('no dobson available')
            dobson_dic=None
        else:
            if wp_out==0:
                wp_out='AD'
            if wp_out==2:
                wp_out='CD'
            if wp_out==4:
                wp_out='AD moon'
                
            if type_out in [0,91,92,93]:
                type_out='DS'
            elif type_out==2:
                type_out='Zb'
            elif type_out in np.arange(3,9):
                type_out='Zc'
            else:
                type_out='XX'
                
            dobson_dic={'TOC' : ozone_out,'Time' : time_out,'WLCode' : wp_out,'ObsType' : type_out,'DecHr' : dechr_out}
    else:
        dobson_dic=None
    #smbclient.reset_connection_cache()
    return dobson_dic