# -*- coding: utf-8 -*-
"""
Created on Mon Jun 20 09:37:31 2016

@author: geddesag
"""
from numpy import *
import numpy as np
import datetime
import ephem
import time


def decdeg2dms(dd):
    """Converts decimal degrees to degrees minutes seconds"""
    is_positive = dd >= 0
    dd = abs(dd)
    minutes,seconds = divmod(dd*3600,60)
    degrees,minutes = divmod(minutes,60)
    degrees = degrees if is_positive else -degrees
    return str(int(degrees))+":"+str(int(minutes))+":"+str(seconds)

def sunzen_ephem(time,Lat,Lon,psurf,temp):
    """Calculates the solar zenith and azimuth angles for a given datetime object (utc), 
    latitude (deg), longitude (deg), psurf(mb), temp(deg c)"""
    observer = ephem.Observer()
    observer.lon = decdeg2dms(Lon)
    observer.lat = decdeg2dms(Lat)
    observer.date = time
 
 
    """If you dont want to consider refraction effects, set observer.pressure to 0"""
    observer.pressure=psurf
    observer.temp=temp
    
    #observer.compute_pressure()
    """We can also include the observer altitude, this is for the geometric difference that you would get
    rather than any difference in pressure, which has already been handled. Its largely irrelevant unless you
    are 10s of km high"""
    
    #observer.altitude=30000 #metres
    
    """set what we are looking at, if we want the moon, we can change to moon = ephem.Moon(observer)
    and then change the subsequent code to call moon.alt or moon.az"""
    
    sun = ephem.Sun(observer)
   # sun.compute(observer)
    alt_atr = float(sun.alt)
    solar_altitude=180.0*alt_atr/pi
    solar_zenith=90.0-solar_altitude
    solar_azimuth=180*float(sun.az)/pi
    return solar_zenith, solar_azimuth
    
def moonzen_ephem(time,Lat,Lon,psurf,temp):
    """Calculates the solar zenith and azimuth angles for a given datetime object (utc), 
    latitude (deg), longitude (deg), psurf(mb), temp(deg c)"""
    observer = ephem.Observer()
    observer.lon = decdeg2dms(Lon)
    observer.lat = decdeg2dms(Lat)
    observer.date = time
 
 
    """If you dont want to consider refraction effects, set observer.pressure to 0"""
    observer.pressure=psurf
    observer.temp=temp
    
    #observer.compute_pressure()
    """We can also include the observer altitude, this is for the geometric difference that you would get
    rather than any difference in pressure, which has already been handled. Its largely irrelevant unless you
    are 10s of km high"""
    
    #observer.altitude=30000 #metres
    
    """set what we are looking at, if we want the moon, we can change to moon = ephem.Moon(observer)
    and then change the subsequent code to call moon.alt or moon.az"""
    
    moon = ephem.Moon(observer)
   # sun.compute(observer)
    alt_atr = float(moon.alt)
    moon_altitude=180.0*alt_atr/pi
    moon_zenith=90.0-moon_altitude
    moon_azimuth=180*float(moon.az)/pi
    return moon_zenith, moon_azimuth
    
    
def format_time(datetime_obj):
    """Quick function that formats a datetime object in to HH:MM:SS string format"""
    out =str('%02d:%02d:%02d' % (int(datetime_obj.hour), int(datetime_obj.minute), int(datetime_obj.second)))
    return out
    
def find_nearest(array,value):
    """Finds the index in array where the array element is closest to the value argument"""
    idx = (np.abs(array-value)).argmin()
    return idx
    
    
def sza_info(lat,lon,utc_offset,psurf,temp):
    """Calculates useful information for Today based on latitude(deg), longitude(deg),
    utc offset, surface pressure (mb) and temperature (deg C)
    """
    time_utc=datetime.datetime.utcnow() #Todays date and time in UTC
   
    time_local=time_utc+datetime.timedelta(hours=utc_offset) #Local date and time
    
    
    times_local=[]
    sza_ref=[] 
    times_utc=[]
    
    """This takes a bit of explaining...Its a loop calculating the sza at different times. However it is 
    not so simple. Because ephem works exclusively in utc time we have to calculate a time array of that
    but it has to represent one day at the location, from midnight to midnight. Thats why I start the array
    at local day at 0:0:0, with a subtraction of the utc_offset, and then iterate over the calculation every day.
    I can probably make this cleaner by using different datetime functions, but because I regularly look at utc
    or local times, it made sense to keep everything as dependent on that than anything else."""
    
    for i in range(86400):
        times_utc.append(datetime.datetime(time_local.year,time_local.month,time_local.day,0,0,0)+datetime.timedelta(seconds=i)-datetime.timedelta(hours=utc_offset))
        sza_ref.append(sunzen_ephem(times_utc[i],lat,lon,psurf,temp)[0])
        times_local.append(times_utc[i]+datetime.timedelta(hours=utc_offset))




    """Now we look through the sza reference array to find the max and minimum values and note there index"""
    sza_time_local=[]
    high_sun_idx=where(array(sza_ref)==min(array(sza_ref)))[0][0]
    low_sun_idx=where(array(sza_ref)==max(array(sza_ref)))[0][0]
    high_sun_sza=sza_ref[high_sun_idx]
    low_sun_sza=sza_ref[low_sun_idx]
    high_sun_time=format_time(times_local[high_sun_idx].time())
    low_sun_time=format_time(times_local[low_sun_idx].time())

    """To find sunrise and sunset is a little trickier. I use a find nearest function to find the index of 
    the array where it is closest to 90 degrees. I do this twice, once before the high sun index (midday)
    and once for the rest of the day. This guarantees we find sunrise and sunset. Currently we define sunrise at 
    SZA = 90 but this could be adjusted easily. """
    sunrise_idx=find_nearest(array(sza_ref[0:high_sun_idx]),90.0)
    sunset_idx=find_nearest(array(sza_ref[high_sun_idx:]),90.0)+high_sun_idx
    sunrise_s=sza_ref[sunrise_idx]
    sunset_s=sza_ref[sunset_idx]
    
    sunrise=times_local[sunrise_idx]
    sunset=times_local[sunset_idx]
    
    
    day_length=str(times_local[sunset_idx]-times_local[sunrise_idx])
    
    """Last little thing, may or may not be useful. if nearest value to 90 is above 91 or below 89, 
    the sun is permanently set or risen. So I return an n/a value for the formatted string."""
    
    if sunrise_s<=89 or sunrise_s>=91.:
        sunrise="n/a"
        day_length="n/a"
    
    if sunset_s<=89 or sunset_s>=91.:
        sunset="n/a"    
        day_length="n/a"
        
    """Here is a quick plot of the sza in local and utc time"""
#    fig=figure(1)
#    ax1=fig.add_subplot(111)
#    ax1.plot(times_local,sza_ref)
#    show()
    
    """ Note the slight wobbles around sza 92, due to refraction, otherwise smooth"""


    return high_sun_time, high_sun_sza, low_sun_time, low_sun_sza, sunrise, sunset, day_length,times_local,sza_ref
    