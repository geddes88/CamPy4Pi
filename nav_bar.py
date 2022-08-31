# -*- coding: utf-8 -*-
"""
Created on Tue Jul  7 15:51:00 2020

@author: uvuser

Navigation bar for use in dashboard 
"""


import dash_bootstrap_components as dbc
def Navbar(label_in):
    navbar = dbc.NavbarSimple(
        brand="Camera Dashboard",
        color="#0f2e6a",
        dark=True,
        fluid=True, 
        brand_style={'justify':"left",'font-family' : 'sans-serif'},

        style={'width' : '100%' ,'margin-left' : '0','margin-top' : '0%','margin-right' : '0%','margin-bottom' : '2%'}
    )
    return navbar