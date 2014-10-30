# Where will the next KC water pipe break?
# 
# Bayesian Statistical Modeling using MCMC
# by: Nick Tomasino

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib import animation
from mpl_toolkits.basemap import Basemap

import pymc as pm
import pandas as pd
import numpy as np
import pydot
import scipy.misc

# from pylab import *
# JSAnimation import available at https://github.com/jakevdp/JSAnimation
# from JSAnimation import IPython_display

#d = pd.read_csv('data/311data') #full set
d = pd.read_csv('data/311data', nrows=100) #limit to 1k rows for testing
d = pd.DataFrame(d, columns=['creation_date', 'closed_date', 'latitude', 'longitude', 'request_type'])
#d.sort(columns=['closed_date'], ascending=True, inplace=True) #do this after filtering to save on cycles

#-1 is the value that find returns if it doesn't find the string
#Filter the Dataset. Remove all rows without 'Water' in their request_type
data = pd.DataFrame
i = 0
for index, row in d.iterrows():
     if(\
       ( str.upper(row['request_type']).find('WATER') != -1 ) and\
       ( row['longitude']  !=0 ) and\
       ( row['latitude' ]  !=0 ) ):            
            row_frame = pd.DataFrame([{'creation_date':row.creation_date, 'closed_date':row.closed_date, 'longitude':row.longitude, 'latitude':row.latitude, 'request_type':row.request_type }])
            if i == 0: 
                data = row_frame
                i+=1
            else: 
                data = data.append(row_frame)

del d #relax d's memory usage
data.sort(columns=['closed_date'], ascending=True, inplace=True)
dt_index = pd.DatetimeIndex(pd.to_datetime(data.closed_date))
data = data.set_index(dt_index)

"""
LAT_BEGIN = 38.75
LAT_END   = 39.5
LON_BEGIN = -96
LON_END   = -95
"""
LAT_BEGIN = 38.8
LAT_END   = 39.35
LON_BEGIN = -94.75
LON_END   = -94.3
N_STEPS   = 3
lat_steps = np.linspace(LAT_BEGIN, LAT_END, N_STEPS)
lon_steps = np.linspace(LON_BEGIN, LON_END, N_STEPS)
"""
the grid variable represents quadrants of latitude and longitude
i.e. a 3x3 grid of squares at every point in time
-95.0 38.75 - grid[0]
-95.5 38.75 - grid[1]
-96.0 38.75 - ...
-95.0 39.125
-95.5 39.125
-96.0 39.125
-95.0 39.5
-95.5 39.5
-96.0 39.5 - grid[8]
"""
#returns the grid indicies for a particular geo point
#N_STEPS-1 => one to compensate the +1 lookahead
def in_square(lat, lon, lat_steps, lon_steps):
    if(lon <= lon_steps[0]):           lon_index = 0
    elif(lon_steps[N_STEPS-1] <= lon): lon_index = N_STEPS-1
    else:
        for x in range(0, N_STEPS - 1):
            if(lon_steps[x] <= lon <= lon_steps[x+1]):  lon_index = x
   
    if(lat <= lat_steps[0]):           lat_index = 0
    elif(lat_steps[N_STEPS-1] <= lat): lat_index = N_STEPS-1            
    else:
        for y in range(0, N_STEPS - 1): 
            if(lat_steps[y] <= lat <= lat_steps[y+1]):  lat_index = y
    return lat_index, lon_index

"""grid_split [] [] / [] [] / [] []
input:  lats - list of latitudes on day d
        lons - list of longitudes on day d
        
output: count for each square
[1] [3] / [2] [1] / [0] [1]
"""
def grid_split_2d(lat_steps, lon_steps, day_data):
    grid = np.zeros((len(lat_steps), len(lon_steps)))
    for i, row in day_data.iterrows():
        lat_index, lon_index = in_square(row.latitude, row.longitude, lat_steps, lon_steps)
        grid[lat_index, lon_index] += 1
    return grid

"""returns a 1d grid in array form"""
def grid_split_1d(lat_steps, lon_steps, day_data):
    grid = np.zeros((len(lat_steps) * len(lon_steps))) #changed to ndarray
    for i, row in day_data.iterrows():
        lat_index, lon_index = in_square(row.latitude, row.longitude, lat_steps, lon_steps)
        grid[(N_STEPS*lat_index) + lon_index] += 1 #could also be len(lon_steps)*lat_index
    return grid

#uses the two functions above to generate a grid dictionary from the data frame
def grid2d_from_data(data, grid2d = dict()):
    for date_index in data.index:    
        if (pd.isnull(date_index)): continue #last few rows are always NaT (Not a Time) :(
        day_data = data[str(date_index)]
        grid2d[date_index] = grid_split_2d(lat_steps, lon_steps, day_data)
    return grid2d

def grid1d_from_data(data, grid1d = dict()):
    for date_index in data.index:    
        if (pd.isnull(date_index)): continue #last few rows are always NaT (Not a Time) :(
        day_data = data[str(date_index)]
        grid1d[date_index] = grid_split_1d(lat_steps, lon_steps, day_data)
    return grid1d

# <codecell>

grid1d = dict()
grid1d = grid1d_from_data(data, grid1d = grid1d)

# <codecell>

grid2d = dict()
grid2d = grid2d_from_data(data, grid2d=grid2d)

# <codecell>

#uniform prior (all ones), either this or discrete_uniform
#unif = pm.DiscreteUniform('unif', 0, 1, size=[1,N_STEPS**2] )
#alpha = unif.random()
alpha = np.ones(N_STEPS**2)
dirich = pm.Dirichlet('dirich', theta = alpha)


#initialize with n equal to the average of the daily count data
total = sum(data.groupby('closed_date')['request_type'].count())
num_elements = len(data.groupby('closed_date'))
avg_calls_per_day = total / num_elements

expon = pm.Exponential('expon', beta = avg_calls_per_day)

#set n = to poison centered on n
#poisson = pm.Poisson('poisson', mu = avg_calls_per_day)
#poisson = pm.Poisson('poisson', mu = avg_calls_per_day, observed=True,
#                      value = [ np.sum(grid1d.values()[i]) for i in range( 0, len(grid1d.values()) ) ])

#"value =" is where the data comes in. adding up all values in the grid to get the poisson count for that day
poisson = pm.Poisson('poisson', mu = expon, observed=True,
                      value = [ np.sum(grid1d.values()[i]) for i in range( 0, len(grid1d.values()) ) ])


#multi = pm.Multinomial('multi', n=poisson, p=dirich, value=grid1d.values(), observed=True) #want to do this
#multi = pm.Multinomial('multi', n=poisson, p=dirich) #works
#multi = pm.Multinomial('multi', n=poisson, p=dirich, value=[0,0,0,0,0,0,1,1,1], observed=True) #works when n == sum(value)

"""
multi = pm.Multinomial('multi', p=dirich, observed=True, 
                        n = [ np.sum(grid1d.values()[i]) for i in range( 0, len(grid1d.values()) ) ] , 
                        value = [ grid1d.values()[i] for i in range( 0, len(grid1d.values()) ) ] )
model = pm.Model([multi, dirich], name = 'model')
"""

#similar to the poisson's 'value =' except each grid get's it's own daily count, 
#instead of adding all of the grid-squares together
multi = pm.Multinomial('multi', p=dirich, observed=True, 
                        n =  poisson, 
                        value = [ grid1d.values()[i] for i in range( 0, len(grid1d.values()) ) ] )
model = pm.Model([multi, dirich, poisson, expon], name = 'model')

# <codecell>

mcmc = pm.MCMC(model)
mcmc.sample(200, 100, 1)

# <codecell>

dirich_samples = mcmc.trace('dirich')[:]
expon_sapmples = mcmc.trace('expon')[:]
#no samples from these last two b/c they're observed
#poisson_samples = mcmc.trace('poisson1')[:]
#multi_samples = mcmc.trace('multi')[:]

# <codecell>

#create new model of with unobserved data using our _posteriors_ from MCMC above
#sampling from this model is like sampling an artificial dataset tuned to our data
poisson1 = pm.Poisson('poisson1', mu = expon)

multi1 = pm.Multinomial('multi1', n=poisson1, p=dirich) #works

model1 = pm.Model([multi1, dirich, poisson1, expon], name = 'model1')

# <codecell>

mcmc1 = pm.MCMC(model1)
mcmc1.sample(200, 100, 1)
#dirich_samples = mcmc.trace('dirich')[:]
#expon_sapmples = mcmc.trace('expon')[:]

#CAN sample from these now b/c they're observed
poisson1_samples = mcmc1.trace('poisson1')[:]
multi1_samples = mcmc1.trace('multi1')[:]

# <codecell>

def add_noise(multi_sample):
    normal = pm.Normal(name='normal', mu=0, tau=1, size=len(multi_sample))
    sample = normal.random() + multi_sample
    return [ 
       round(sample[i]) 
       if round(sample[i]) > 0
       else 0.0
       for i in range(len(sample))
    ]

def condition_dict_grid(grid, sector, num_calls):
    ret = list()
    for sectors in grid.itervalues():
        #sector we got a call in matches the number of calls we're looking for
        if sectors[sector] == num_calls: 
            ret.append(sectors)
    return ret

#TODO change grid1d from a dict to a list and combine these functions
#returns a subset of grid with rows whose sector==num_calls 
def condition_grid(grid, sector, num_calls):
    ret = list()
    for sectors in grid:
        #sector we got a call in matches the number of calls we're looking for
        if sectors[sector] == num_calls: 
            ret.append(sectors)
    return ret

"""generates an artificial datapoint 
(by using the posteriors from mcmc for dirichlet and exponential)
multi1.random()

min support - minimum number of points needed for a reasonable prediction to be made
if there aren't enough points in the real dataset, then we sample from
the unobserved_multinomial to augment our artificial dataset"""

def augment_grid(sector, num_calls, samples_per_iter = 100, min_support = 20, grid_real = list(), MAX_SAMPLES=20000):
    grid1d_artificial = []
    conditioned_grid_real = condition_grid( grid_real, sector, num_calls)
    if len(conditioned_grid_real) >= min_support:
        return conditioned_grid_real
    else:
        iters = 0
        while ( len(grid1d_artificial) < (min_support - len(conditioned_grid_real)) ) and (iters*samples_per_iter < MAX_SAMPLES):
            artificial_samples = [add_noise(multi1.random()) for i in range(samples_per_iter)]
            iters+=1
            if np.mod(iters, 100) == 0: print 'sample iter 100 x ', iters
            grid1d_artificial.extend(  condition_grid( artificial_samples, sector, num_calls )  )
    return grid1d_artificial

"""
GRIDSQUARES -> LATS, LONS
.--> (x) == lon_steps
| 
v (y) == lat_steps

lats and lons were hashed into grid squares via the gridsplit1d and in_square functions
grid[(N_STEPS*lat_index) + lon_index] += 1
"""

lat_stepsize = np.abs(lat_steps[0] - lat_steps[1])
lon_stepsize = np.abs(lon_steps[0] - lon_steps[1])

lat_sd = lat_stepsize**2 if lat_stepsize < 1 else np.sqrt(lat_stepsize)
lon_sd = lon_stepsize**2 if lon_stepsize < 1 else np.sqrt(lon_stepsize)

def grid1dToLatsLons_plot(grid):
    lats = []; lons = [];
    for i in range( len( grid) ):
        for j in range( len( grid[i] ) ):
            num_calls = grid[i][j]
            if num_calls == 0: continue
            lat_step = np.mod(j, N_STEPS)
            lon_step = j / N_STEPS
            
            lat_mean = lat_steps[lat_step] 
            lon_mean = lon_steps[lon_step]               
            #lats.append(lat_mean) #works
            #lons.append(lon_mean)
            
            lats.extend(np.random.normal(loc=lat_mean, scale=lat_sd, size = num_calls )) 
            lons.extend(np.random.normal(loc=lon_mean, scale=lat_sd, size = num_calls )) 
    return lats, lons

#for use with plt.scatter(). 
def grid1dToLatsLonsSize_scatter(grid):
    lats = []; lons = []; sizes=[]
    for i in range( len( grid) ):
        for j in range( len( grid[i] ) ):
            num_calls = grid[i][j]
            if num_calls == 0: continue
            lat_step = np.mod(j, N_STEPS)
            lon_step = j / N_STEPS
            
            #if (lat_step == 0 and lon_step==0):
            lat_mean = lat_steps[lat_step]
            lon_mean = lon_steps[lon_step]
            
            lats.append(np.random.normal(loc=lat_mean, scale=lat_sd ))
            lons.append(np.random.normal(loc=lon_mean, scale=lat_sd ))
            sizes.append(num_calls)
    return lats, lons, sizes

geo_map = Basemap(projection='merc', lat_0=39, lon_0=-94.5,
    resolution = 'l', area_thresh = 3000.0,
    llcrnrlon=-95, llcrnrlat=38.7,
    urcrnrlon=-94, urcrnrlat=39.6)

def generate_samples(sector = 1, num_calls = 1, grid = grid1d):
    conditioned_grid = condition_dict_grid(grid1d, sector=sector, num_calls = num_calls)
    cgrid_upsampled = augment_grid( sector, num_calls, grid_real= conditioned_grid)

    lats, lons, sizes = grid1dToLatsLonsSize_scatter(conditioned_grid)
    xpts, ypts = geo_map(lons, lats)

    lats, lons, sizes = grid1dToLatsLonsSize_scatter(cgrid_upsampled)
    xpts_sampled, ypts_sampled = geo_map(lons, lats)
    return xpts, ypts, xpts_sampled, ypts_sampled

def inc_dict(dictionary, key):
    if dictionary.__contains__(key):     dictionary[key] += 1
    else: dictionary[key] = 1
    
call_dict={}
for i in range(N_STEPS**2): call_dict[i]=0
call_dict[1]=0 #keeps track of sector and num_calls on the grid

xpts, ypts, xpts_sampled, ypts_sampled = generate_samples(sector=1, num_calls=2, grid=grid1d)

"""
Graph a heatmap of the data by sector
May be conditioned on getting an exact number of calls in a sector
"""

from matplotlib.widgets import Button

plt.figure(figsize=(24,12))
plt.axes().set_axis_bgcolor('grey')
img=mpimg.imread('kcmo1.png')
left = np.min(xpts_sampled); right = np.max(xpts_sampled); bottom = np.min(ypts_sampled); top = np.max(ypts_sampled)
imgplot = plt.imshow(img, extent=[left, right, bottom, top])
size = ( np.abs(right-left) + np.abs(top-bottom) ) / (2*20)
real_points = plt.axes().scatter(x=xpts, y=ypts, alpha=.4, s=size, c='cyan', label='real data')
sampled_points = plt.axes().scatter(x=xpts_sampled, y=ypts_sampled, alpha=.1, s=size, c = 'violet', label='sampled data')
plt.axes().legend(('real data', 'sampled data'), loc='lower right', markerscale=.2, framealpha=.7 )

#Call_Dictionary Display axis
s = ''
call_dict_axis = plt.axes([0.8, .615, 0.1, 0.3], alpha = .2)
call_dict_text = call_dict_axis.text(.7, .415, s)

#Reset Button
#*rect* = [left, bottom, width, height] 
reset_axis = plt.axes([0.8, 0.15, 0.1, 0.04], alpha = .2) #TODO alpha here not working 
reset_button = Button(ax=reset_axis, label='Reset', color='lightblue' , hovercolor='0.975') 

#Button: add to square 1
axis1 = plt.axes([0.2, 0.35, 0.1, 0.04], alpha = .2) 
axis2 = plt.axes([0.2, 0.55, 0.1, 0.04], alpha = .2) 
button1 = Button(ax=axis1, label='Inc1', color='lightblue' , hovercolor='0.975') 
button2 = Button(ax=axis2, label='Inc2', color='lightblue' , hovercolor='0.975') 

def clear_points():
    global real_points
    global sampled_points
    try:    sampled_points.remove()
    except: pass
    try:    real_points.remove()
    except: pass

def reset(event):                                                               
    global call_dict
    clear_points()
    call_dict.clear()
    for i in range(N_STEPS**2): call_dict[i]=0

def Inc1(event):                                                               
    global call_dict
    global call_dict_text
    global real_points
    global sampled_points
    clear_points()
    inc_dict(call_dict, key=1)
    xpts, ypts, xpts_sampled, ypts_sampled = generate_samples(sector=3, num_calls=call_dict[1], grid=grid1d)
    real_points    = plt.axes().scatter(x=xpts, y=ypts, alpha=.4, s=size, c='cyan', label='real data')
    sampled_points = plt.axes().scatter(x=xpts_sampled, y=ypts_sampled, alpha=.1, s=size, c = 'violet', label='sampled data')
    try:    call_dict_text.remove()    
    except: pass
    s = ''
    for i in xrange(N_STEPS**2):
        s += str(i) + ': ' + str(call_dict[i]) + '\n'
    call_dict_axis = plt.axes([0.8, .615, 0.1, 0.3], alpha = .2)
    call_dict_text = call_dict_axis.text(.7, .415, s)

def Inc2(event):                                                               
    global call_dict
    global call_dict_text
    global real_points
    global sampled_points
    clear_points()
    inc_dict(call_dict, key=2)
    xpts, ypts, xpts_sampled, ypts_sampled = generate_samples(sector=3, num_calls=call_dict[2], grid=grid1d)
    real_points    = plt.axes().scatter(x=xpts, y=ypts, alpha=.4, s=size, c='cyan', label='real data')
    sampled_points = plt.axes().scatter(x=xpts_sampled, y=ypts_sampled, alpha=.1, s=size, c = 'violet', label='sampled data')
    try:    call_dict_text.remove()    
    except: pass
    s = ''
    for i in xrange(N_STEPS**2):
        s += str(i) + ': ' + str(call_dict[i]) + '\n'
    call_dict_axis = plt.axes([0.8, .615, 0.1, 0.3], alpha = .2)
    call_dict_text = call_dict_axis.text(.7, .415, s)

reset_button.on_clicked(reset)
button1.on_clicked(Inc1)
button2.on_clicked(Inc2)

plt.show()
