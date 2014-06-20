# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons                     
from mpl_toolkits.basemap import Basemap
import numpy as np
from pylab import *                                                             

d = pd.read_csv('data/311data', nrows=1000) #limit to 1k rows for testing
d = pd.DataFrame(d,columns=['creation_date', 'closed_date', 'latitude', 'longitude', 'request_type'])
#d.sort(columns=['closed_date'], ascending=True, inplace=True)

#-1 is the value that find returns if it doesn't find the string
#Filter the Dataset. Remove all rows without 'Water' in their request_type
data = pd.DataFrame
i = 0
for index, row in d.iterrows():
     if(\
       ( str.upper(row['request_type']).find('WATER') != -1 ) and\
       ( row['longitude']  !=0 ) and\
       ( row['latitude' ]  !=0 ) ):            
            row_frame = pd.DataFrame([{'creation_date':row.creation_date,'closed_date':row.closed_date, 'longitude':row.longitude, 'latitude':row.latitude,'request_type':row.request_type }])
            if i == 0: 
                data = row_frame
                i+=1
            else: 
                data = data.append(row_frame)

del d #relax d's memory usage
data.sort(columns=['closed_date'], ascending=True, inplace=True)
#d=0 
#data.sort_index(by='closed_date', inplace=True, ascending=True)

#assumes the data is sorted by closed_date
#assumes a 'data' DataFrame with latitude and longitude and closed_date columns
#time is the percentage through the dataset
#global is used because this function is called from an implicit plot function
def getLatsAndLongsByTimePercentage(time, num_points = -12):
    global data
    if (num_points == -12): num_points = len(data) / math.sqrt(len(data))
    if (time > 1.0) or (time < 0.0): return 0
    start_index = int( len(data) * time )
    stop_index  = int( start_index + num_points )
    start_date = pd.DataFrame(data, columns=['closed_date']).values[start_index]
    stop_date  = pd.DataFrame(data, columns=['closed_date']).values[stop_index]
    lats = pd.DataFrame(data, columns=['latitude']).values[start_index : stop_index]
    longs= pd.DataFrame(data, columns=['longitude']).values[start_index : stop_index]
    return lats,longs,start_date,stop_date

# HEATMAP from http://matplotlib.org/basemap/users/examples.html

plt.figure(figsize=(24,12))
axis1 = subplot(111)
map = Basemap(projection='merc', lat_0=39, lon_0=-94,
    resolution = 'c', area_thresh = 100000.0,
    llcrnrlon=-95, llcrnrlat=38.75,
    urcrnrlon=-94, urcrnrlat=39.5)

axis1.text(0.95, 0.01, 'start_date and stop_date',
    verticalalignment='bottom', horizontalalignment='right',
    transform=axis1.transAxes, color='cyan', fontsize=15)

#map.drawcounties()

t0 = .5
slider_min = 0.0
slider_max  = 1.0

lats_lons = getLatsAndLongsByTimePercentage(t0)
lats = lats_lons[0]
lons = lats_lons[1]
x,y = map(lons, lats)
map.plot(x, y, 'ro',fillstyle='none', markersize=8)

#---Slider Code--- 
subplots_adjust(left=0.25, bottom=0.25)                                         
slider_color = 'lightgoldenrodyellow'                                                
slider_dimens  = axes([0.25, 0.15, 0.65, 0.03], axisbg=slider_color)                         

slider_time = Slider(slider_dimens, 'Time', slider_min, slider_max, valinit=t0)
slider_time.label='slider label'

def update(val):
    subplot(111)
    plt.cla()
    time_percentage = val
    lats_lons = getLatsAndLongsByTimePercentage(time_percentage)
    lats = lats_lons[0]
    lons = lats_lons[1]
    start_date = lats_lons[2]
    stop_date  = lats_lons[3]
    axis1.text(0.95, 0.01, start_date+'  '+stop_date,
        verticalalignment='bottom', horizontalalignment='right',
        transform=axis1.transAxes, color='cyan', fontsize=15)
    x,y = map(lons, lats)
    map.plot(x, y, 'ro',fillstyle='none', markersize=8)
slider_time.on_changed(update)

resetax = axes([0.8, 0.025, 0.1, 0.04])                                         
button = Button(resetax, 'Draw Borders', color=slider_color, hovercolor='0.975')            
def reset(event):                                                               
    map.drawstates()
    map.drawcounties()
    map.drawrivers()
button.on_clicked(reset)

plt.show()
