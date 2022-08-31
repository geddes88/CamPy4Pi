import datetime

import dash
import dash_core_components as dcc
import dash_dangerously_set_inner_html
import dash_html_components as html
import pandas as pd
from dash.dependencies import Output,Input
from dash.exceptions import PreventUpdate
from library import *

import json
import random
import os
import nav_bar
from PIL import Image
from io import BytesIO
import requests
import dash_bootstrap_components as dbc
import sys
import time

"""Lets build the layout"""


print(dcc.__version__) # 0.6.0 or above is required

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__,)

tabulated_data=pd.DataFrame()


app.layout = html.Div([
    dcc.Interval(id='live_clock', interval=60 * 1000, n_intervals=0),
    dcc.Interval(id='clock_clock', interval=5 * 1000, n_intervals=0),

    nav_bar.Navbar('Camera 1'),
    #html.H1('Allsky Camera Dashboard',style={'color': '#0078bc', 'justify': 'center', 'font-family': 'sans-serif'}),
    html.Div([
        html.Div([
            html.Div([
                html.H6('Local Time (NZST)', id='local_time_label',
                        style={'color': '#0078bc', 'justify': 'center', 'font-family': 'sans-serif'}),
                html.H6('HH:MM', id='local_time',
                        style={'color': "#0f2e6a", 'justify': 'center', 'font-family': 'sans-serif'}),
                html.H6('Obs. Time (NZST)', id='obs_time_label',
                        style={'color': '#0078bc', 'justify': 'center', 'font-family': 'sans-serif'}),
                html.H6('HH:MM', id='obs_time',
                        style={'color': "#0f2e6a", 'justify': 'center', 'font-family': 'sans-serif'}),
                html.H6('Cloud Fraction', id='cloud_frac_label',
                        style={'color': '#0078bc', 'justify': 'center', 'font-family': 'sans-serif'}),
                html.H6('XX.XX', id='cloud_fraction',
                        style={'color': "#0f2e6a", 'justify': 'center', 'font-family': 'sans-serif'}),
            ], className='four columns', style={'textAlign': 'center'}),

            html.Div([
                html.H6('Solar Zenith Angle', id='sza_label',
                        style={'color': '#0078bc', 'justify': 'center', 'font-family': 'sans-serif'}),
                html.H6('XXX.XX', id='sza_value',
                        style={'color': "#0f2e6a", 'justify': 'center', 'font-family': 'sans-serif'}),
                html.H6('Azimuth Angle', id='saa_label',
                        style={'color': '#0078bc', 'justify': 'center', 'font-family': 'sans-serif'}),
                html.H6('XXX.XX', id='saz_value',
                        style={'color': "#0f2e6a", 'justify': 'center', 'font-family': 'sans-serif'}),
                html.H6('Camera Temp.', id='cam_temp_label',
                        style={'color': '#0078bc', 'justify': 'center', 'font-family': 'sans-serif'}),
                html.H6('XX.XX', id='cam_temp',
                        style={'color': "#0f2e6a", 'justify': 'center', 'font-family': 'sans-serif'}),
            ], className='four columns', style={'textAlign': 'center'}),

            html.Div([
                html.H6('Sunrise (-MM:SS)', id='sunrise_time_label',
                        style={'color': '#0078bc', 'justify': 'center', 'font-family': 'sans-serif'}),
                html.H6('HH:MM (+MM:SS)', id='sunrise_time',
                        style={'color': "#0f2e6a", 'justify': 'center', 'font-family': 'sans-serif'}),
                html.H6('Sunset  (-MM:SS)', id='sunset_time_label',
                        style={'color': '#0078bc', 'justify': 'center', 'font-family': 'sans-serif'}),
                html.H6('HH:MM (+MM:SS)', id='sunset_time',
                        style={'color': "#0f2e6a", 'justify': 'center', 'font-family': 'sans-serif'}),
                html.H6('Sun Out Flag', id='sun_flag_label',
                        style={'color': '#0078bc', 'justify': 'center', 'font-family': 'sans-serif'}),
                html.H6('T/F', id='sun_out_flag',
                        style={'color': "#0f2e6a", 'justify': 'center', 'font-family': 'sans-serif'}),
            ], className='four columns', style={'textAlign': 'center'}),

        ], style={'margin-bottom': '30%', 'margin-left': '2%', 'margin-right': '2%'}),
        html.Div([
            dcc.Graph(id='cloud_graph', style={'width': '100%', 'margin-top': '0%'}, figure={
                'layout': {'autosize': True, 'yaxis': {'title': 'hello'},
                           'yaxis2': {'overlaying': 'y', 'side': 'right'}}})

        ]),

    ],className='one-half column'),
    html.Div([
        html.Img(src=app.get_asset_url("Sun.png"),id='allsky_image',style={'height' : '45%','width' : '90%'}),
        html.Img(src=app.get_asset_url("Cloudy.png"),id='cloud_fraction_image',style={'height' : '45%','width' : '90%'}),
    ],className='one-half column'),
])

"""Live Feed Callbacks"""


@app.callback([Output('local_time', 'children'), Output('sza_value', 'children'), Output('saz_value', 'children'),
               Output('sunrise_time', 'children'), Output('sunset_time', 'children')],
              [Input('clock_clock', 'n_intervals')])
def update_time(n):
    time_now = datetime.datetime.utcnow() + datetime.timedelta(hours=12)
    if time_now.hour == 1 and time_now.minute == 0:
        print('restarting')
        time.sleep(60)
        restart()
    sza, saz = sunzen_ephem(time_now - datetime.timedelta(hours=12), -45.043, 169.68, 0, 0, 370)
    sunrise, sunset = sunrise_sunset(time_now - datetime.timedelta(hours=12), -45.043, 169.68, 0, 0, 370)
    sunrise += datetime.timedelta(hours=12)
    sunset += datetime.timedelta(hours=12)
    sunrise_y, sunset_y = sunrise_sunset(time_now - datetime.timedelta(hours=36), -45.043, 169.68, 0, 0, 370)
    sunrise_y += datetime.timedelta(hours=12)
    sunset_y += datetime.timedelta(hours=12)
    sunrise_diff = sunrise - sunrise_y
    sunset_diff = sunset - sunset_y

    sunrise_diff_ts = sunrise_diff.total_seconds() - 24 * 60 * 60
    sunset_diff_ts = sunset_diff.total_seconds() - 24 * 60 * 60

    ss_prefix = '+'
    if sunset_diff_ts < 0:
        ss_prefix = '-'
    sr_prefix = '+'
    if sunrise_diff_ts < 0:
        sr_prefix = '-'

    sunset_diff_ts = np.abs(sunset_diff_ts)
    sunrise_diff_ts = np.abs(sunrise_diff_ts)

    sunrise_diff_mins = np.floor(sunrise_diff_ts / 60)
    sunrise_seconds = int(sunrise_diff_ts - 60 * sunrise_diff_mins)

    sunset_diff_mins = np.floor(sunset_diff_ts / 60)
    sunset_seconds = int(sunset_diff_ts - 60 * sunset_diff_mins)

    sunrise_string = datetime.datetime.strftime(sunrise, '%H:%M') + ' (' + sr_prefix + '%02d:%02d)' % (
    sunrise_diff_mins, sunrise_seconds)
    sunset_string = datetime.datetime.strftime(sunset, '%H:%M') + ' (' + ss_prefix + '%02d:%02d)' % (
    sunset_diff_mins, sunset_seconds)

    return datetime.datetime.strftime(time_now, '%H:%M'), '%.2f' % sza, '%.2f' % saz, sunrise_string, sunset_string

@app.callback([Output('obs_time','children'),Output('cloud_fraction','children'),Output('cam_temp','children'),Output('sun_out_flag','children'),Output('cloud_graph','figure')], [Input('live_clock', 'n_intervals')])
def update_allsky_data(n):
    """Update all the values, randoms for now"""
    temperature=random.uniform(-20,100)
    sun_out_flag=bool(random.getrandbits(1))
    cloud_fraction=random.uniform(0,1)
    recent_time=datetime.datetime.utcnow()+datetime.timedelta(hours=12)
    times,clouds=get_last_24_hours()
    times.append(recent_time)
    clouds.append(cloud_fraction)
    # data=pd.DataFrame()
    # data['times']=times
    # data['cloud']=clouds
    layouts={}
    layouts['autosize'] = True
    layouts['yaxis']={'title' : 'Cloud Fraction'}
    data_out_plot=[]
    data_out_plot.append({'x' :times,'y': clouds,'name' : 'Cloud Fraction', 'type' : 'lines'})
    return datetime.datetime.strftime(recent_time,'%H:%M'),cloud_fraction,temperature,str(sun_out_flag),{'data': data_out_plot, 'layout': layouts}

def get_last_24_hours():
    start_time=(datetime.datetime.utcnow()-datetime.timedelta(hours=12)).replace(second=0,microsecond=0)
    end_time=(datetime.datetime.utcnow()+datetime.timedelta(hours=12,minutes=1)).replace(second=0,microsecond=0)
    times=[start_time+datetime.timedelta(minutes=10*x) for x in range(24*6)]
    clouds=[random.uniform(0,1) for x in times]

    return times,clouds
if __name__ == '__main__':
    app.run_server(debug=True)
