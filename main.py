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
                html.H6('White Cloud Fraction', id='cloud_frac_label',
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
                html.H6('Grey/Uncertain Fraction', id='cam_temp_label',
                        style={'color': '#0078bc', 'justify': 'center', 'font-family': 'sans-serif'}),
                html.H6('XX.XX', id='grey_fraction',
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
                html.H6('Sun Saturation', id='sun_flag_label',
                        style={'color': '#0078bc', 'justify': 'center', 'font-family': 'sans-serif'}),
                html.H6('T/F', id='sun_sat',
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
        html.Img(src=app.get_asset_url("Sun.png"),id='allsky_image',style={'height' : 'auto','width' : '70%'}),
        html.Img(src=app.get_asset_url("Cloudy.png"),id='cloud_image',style={'height' : 'auto','width' : '70%'}),
    ],className='one-half column'),
])

"""Live Feed Callbacks"""

def restart():
    import sys
    print("argv was",sys.argv)
    print("sys.executable was", sys.executable)
    print("restart now")

    import os
    os.execv(sys.executable, ['python'] + sys.argv)

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

@app.callback([Output('obs_time','children'),Output('cloud_fraction','children'),Output('grey_fraction','children'),Output('sun_sat','children'),Output('cloud_graph','figure'),Output('allsky_image','src'),Output('cloud_image','src')], [Input('live_clock', 'n_intervals')])
def update_allsky_data(n):
    """Update all the values, randoms for now"""

    recent_time=datetime.datetime.utcnow()+datetime.timedelta(hours=12)
    data=read_todays_data(recent_time)
    if recent_time-data['times'][data.index.stop-1]<datetime.timedelta(minutes=10):
        cloud_fraction=data['cloud_fraction'][data.index.stop-1]
        sun_out_flag=data['sun_flag'][data.index.stop-1]
        grey_fraction=data['grey_fraction'][data.index.stop-1]
        allsky_image=get_as_base64_file(data['image_names'][data.index.stop-1])
        cloud_image=get_as_base64_file(data['cloud_names'][data.index.stop-1])
        allsky_src='data:image/png;base64,{}'.format(allsky_image.decode())
        cloud_src='data:image/png;base64,{}'.format(cloud_image.decode())

    else:
        cloud_fraction=np.nan
        sun_out_flag=np.nan
        grey_fraction=np.nan
        allsky_src=None
        cloud_src=None
        
    

    layouts={}
    layouts['autosize'] = True
    layouts['yaxis']={'title' : 'Cloud Fraction'}
    data_out_plot=[]
    data_out_plot.append({'x' :data['times'],'y': data['cloud_fraction'],'name' : 'White Cloud Fraction', 'type' : 'lines'})
    data_out_plot.append({'x' :data['times'],'y': data['grey_fraction'],'name' : 'Grey/Uncertain Fraction', 'type' : 'lines'})
    data_out_plot.append({'x' :data['times'],'y': data['grey_fraction']+data['cloud_fraction'],'name' : 'Total Cloud Fraction', 'type' : 'lines'})


    return datetime.datetime.strftime(data['times'][data.index.stop-1],'%H:%M'),'%.2f' % cloud_fraction,'%.2f' % grey_fraction,'%.2f' % sun_out_flag,{'data': data_out_plot, 'layout': layouts},allsky_src,cloud_src


if __name__ == '__main__':
    app.run_server(host='0.0.0.0',port=8050,debug=True)
