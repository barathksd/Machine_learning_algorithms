# -*- coding: utf-8 -*-
"""
Created on Sun Apr 11 16:45:32 2021

@author: Lenovo
"""

import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import cv2
import os
import sys
import pandas as pd

file = 'c:/users/Lenovo/Desktop/data/intro_bees.csv'
df = pd.read_csv(file)
df = df.groupby(['State','ANSI','Affected by','Year','state_code'])['Pct of Colonies Impacted'].mean()
df = df.reset_index()
years = df['Year'].unique()
affectedby = df['Affected by'].unique()
print(df[:5])


app = dash.Dash(__name__, prevent_initial_callbacks=True)
app.layout = html.Div([
    
    html.H1('Dashboard in python',style={'text-align':'center'}),
    html.Div([
    dcc.Dropdown(id='year_dd',
                 options=[
                     {'label':str(i),'value':i} for i in years
                     ],
                 multi=False,
                 value=np.min(years),
                 style={'width':'40%'}),
    dcc.Dropdown(id='affectedby_dd',
                 options=[
                     {'label':i,'value':i} for i in affectedby
                     ],
                 multi=False,
                 value=affectedby[0],
                 style={'width':'40%'})]),
    html.Div(id='output_container',children=[]),
    html.Br(),
    dcc.Graph(id='my_bee_map',figure={})
    ])

# connect plotly graphs and dash components
@app.callback([
     Output(component_id='output_container',component_property='children'),
     Output(component_id='my_bee_map',component_property='figure')
     ],
    [
     Input(component_id='year_dd',component_property='value'),
     Input(component_id='affectedby_dd',component_property='value')
     ]
    )

def update_graph(year,afby='Pesticides'):
    
    #print('Year chosen : ',year)
    container = f'Year selected : {year}'
    
    dfc = df.copy()[df['Year']==year]
    dfc = dfc[dfc['Affected by']==afby]
    
    #plotly express
    fig = px.choropleth(
        data_frame=dfc,
        locationmode='USA-states',
        locations='state_code',
        scope='usa',
        color='Pct of Colonies Impacted',
        hover_data=['State','Pct of Colonies Impacted','Year','Affected by'],
        color_continuous_scale=px.colors.sequential.YlOrRd,
        labels={'Pct of Colonies Impacted':'% of bee colonies'},
        template='plotly_dark'
        )
    
    #plotly graph objects (GO)

    return container, fig

if __name__ == '__main__':
    app.run_server(debug=True)