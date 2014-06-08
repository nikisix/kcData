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

#-1 is the value that find returns if it doesn't find the string
#Filter the Dataset. Remove all rows without 'Water' in their request_type
data = pd.DataFrame
i = 0
for index, row in d.iterrows():
     if(\
       ( row['request_type'].find('Water') != -1 ) and\
       ( row['longitude']  !=0 ) and\
       ( row['latitude' ]  !=0 ) ):            
            row_frame = pd.DataFrame([{'creation_date':row.creation_date,'closed_date':row.closed_date, 'longitude':row.longitude, 'latitude':row.latitude,'request_type':row.request_type }])
            if i == 0: 
                data = row_frame
                i+=1
            else: 
                data = data.append(row_frame)

#assumes the data is sorted by date
#assumes a 'data' DataFrame with latitude and longitude columns
#time is a percentage through the dataset
def getLatsAndLongsByTimePercentage(time, num_points = -12):
    global data
    if (num_points == -12): num_points = len(data) / math.sqrt(len(data))
    if (time > 1.0) or (time < 0.0): return 0
    start_index = len(d) * time
    stop_index  = start_index + num_points
    lats = pd.DataFrame(d, columns=['latitude']).values[start_index : stop_index]
    longs= pd.DataFrame(d, columns=['longitude']).values[start_index : stop_index]
    return lats,longs

# HEATMAP from http://matplotlib.org/basemap/users/examples.html

p = plt.figure(figsize=(24,12))
map = Basemap(projection='merc', lat_0=39, lon_0=-94,
    resolution = 'l', area_thresh = 3000.0,
    llcrnrlon=-95, llcrnrlat=38.75,
    urcrnrlon=-94, urcrnrlat=39.5)
 
#map.drawcounties()
#map.drawstates()
#map.drawrivers()

t0 = .5
slider_min = 0.0
slider_max  = 1.0

lats_lons = getLatsAndLongsByTimePercentage(t0)
lats = lats_lons[0]
lons = lats_lons[1]
x,y = map(lons, lats)
map.plot(x, y, 'ro',fillstyle='none', markersize=5)

#---Slider Code--- 
slider = subplot(111)                                                               
subplots_adjust(left=0.25, bottom=0.25)                                         
slider_color = 'lightgoldenrodyellow'                                                
slider_dimens  = axes([0.25, 0.15, 0.65, 0.03], axisbg=slider_color)                         

slider_time = Slider(slider_dimens, 'Time', slider_min, slider_max, valinit=t0)

def update(val):
    time_percentage = val
    lats_lons = getLatsAndLongsByTimePercentage(time_percentage)
    lats = lats_lons[0]
    lons = lats_lons[1]
    x,y = map(lons, lats)
    map.plot(x, y, 'ro',fillstyle='none', markersize=5)
    #plt.draw()
    #draw()
    p.draw(map)
slider_time.on_changed(update)

plt.show()
