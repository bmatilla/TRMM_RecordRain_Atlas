#!/usr/bin/env python
# coding=utf-8
"""
"""

__author__ = 'Brian Matilla'
__version__= '1.0.0'
__maintainer__= 'Brian Matilla'
__email__= 'bmatilla@rsmas.miami.edu'

#Load lots of dependencies
import argparse
import numpy as np
import xarray as xr
from cdo import *
cdo= Cdo()
import datetime as dt
from scipy.misc import bytescale
from numpngw import write_png #This can be downloaded from (https://pypi.python.org/pypi/numpngw)
import PIL
from PIL import Image
import matplotlib as mpl
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import cartopy.feature as cfeature
from dask.diagnostics import ProgressBar

#First, build the parser for the moving average length (cdo) and grid to use.

def get_args():
    '''This function is the one that obtains and parses the arguments passed through the command
    line'''
    
    if __name__ == '__main__':
        
        parser = argparse.ArgumentParser(description='Script to take user-specified moving average length and selected TRMM grid. The outputs will be a background B/W .png file that contains the data (precip max and date & time of occurrence), a duplicate, colored .gif image for the browser, and the associated .npy arrays.',
                                         epilog= "Example use case on the command line: python TRMM_Atlas_1998_2016.py "
                                         "-mvavg moving_avg_length "
                                         "-grid select_grid " )
        
        parser.add_argument('-mvavg', '--moving_avg_length', nargs=1, type=int,
                            help= 'Specified length to calculate the x-day moving average of record events, expressed as an'
                            'integer. X corresponds to the length of days in the moving average',
                            required= True)
        
        parser.add_argument('-grid', '--select_grid', nargs=1, type=float,
                            help= 'Specified TRMM grid to calculate records over. Current choices are 0.25, 1, 2, or 4 degrees.',
                            required= True)
        
        args= parser.parse_args()
        
        moving_avg_length= args.moving_avg_length
        select_grid= args.select_grid
        
        if args.moving_avg_length:
            mvavg_size= args.moving_avg_length[0]
        
        if args.select_grid:
            grid_size= args.select_grid[0]
            
        return moving_avg_length, select_grid
    
moving_avg_length, select_grid= get_args()
            
#Dataset links as they are server-side.
if select_grid[0] == 0.25:
    ds= '/data2/bmatilla/TRMM_3B42/V7_PcpOnly/products_3hrtotals/TRMM_0p25deg_1998_2016.nc'
    print("Be patient, this dataset could take a considerable amount of time to process.")
elif select_grid[0] == 1:
    ds= '/data2/bmatilla/TRMM_3B42/V7_PcpOnly/products_3hrtotals/TRMM_1deg_1998_2016.nc'
elif select_grid[0] == 2:
    ds= '/data2/bmatilla/TRMM_3B42/V7_PcpOnly/products_3hrtotals/TRMM_2deg_1998_2016.nc'
elif select_grid[0] == 4:
    #ds= '/data2/bmatilla/TRMM_3B42/V7_PcpOnly/products_3hrtotals/TRMM_4deg_1998_2016.nc'
    ds= 'TRMM_4deg_1998_2016.nc'
else:
    raise ValueError("This grid is not available for use. Current options are 0.25, 1, 2, or 4-degree grids.")

ds_vars= xr.open_dataset(ds, decode_times=True, chunks=2000) #Chunks of 2000 I've found are safe enough to load into memory.

#Read in the lon,lat, and time variables from dataset.

lons = ds_vars.variables['lon'][:]
lats = ds_vars.variables['lat'][:] 
times = ds_vars.variables['time'][:]

#Run the cdo-based convolution based on the specified moving average length while converting the output to a usable NumPy array.
print("Dataset loaded. Performing moving average calculation.")
with ProgressBar():
    pcp_mvavg= cdo.runsum(((8*moving_avg_length[0])+1), input=ds_vars, returnCdf=True).variables['pcp'][:]

#Calculate the precip max for all grid cells.
pcp_max= ([])
pcp_max= pcp_mvavg.max(axis=0)

print("The minimum precipitation value is " + str(pcp_max.min()) + " mm.")
print("The maximum precipitation value is " + str(pcp_max.max()) + " mm.")

#Now extract the time of record.

pcp_timmax= ([])
pcp_timmax= (pcp_mvavg.argmax(axis=0)*3) #Multiply by 3 hours to match TRMM output.
pcp_timmax_flat= pcp_timmax.flatten() #We need to flatten the time of max array to run it through a loop later in the script.

#Establish arrays for the year, month, day, and hour of record.
year_of_max= ([])
month_of_max= ([])
day_of_max= ([])
hour_of_max= ([])

#Stash the respective data into the arrays with a loop.
for i in pcp_timmax_flat:
    dtg= dt.datetime(1998,1,1,0,0,0)
    dtg2= dt.timedelta(0,3600)*i
    pcp_timmax_validtime= dtg + dtg2
    year_of_max= np.append(year_of_max, pcp_timmax_validtime.year)
    month_of_max= np.append(month_of_max, pcp_timmax_validtime.month)
    day_of_max= np.append(day_of_max, pcp_timmax_validtime.day)
    hour_of_max= np.append(hour_of_max, pcp_timmax_validtime.hour)
    
#Reshape the arrays to the same shape as the precip max array for uniformity.
year_of_max= year_of_max.reshape(pcp_max.shape)
month_of_max= month_of_max.reshape(pcp_max.shape)
day_of_max= day_of_max.reshape(pcp_max.shape)
hour_of_max= hour_of_max.reshape(pcp_max.shape)

#Now, dump the outputs to reusable, reloadable .npy files for later use:

pcp_max.dump("pcp_max_"+str(select_grid[0])+"deg_"+str(moving_avg_length[0])+"day.npy")
pcp_timmax.dump("pcp_timmax_"+str(select_grid[0])+"deg_"+str(moving_avg_length[0])+"day.npy")
year_of_max.dump("year_of_max_"+str(select_grid[0])+"deg_"+str(moving_avg_length[0])+"day.npy")
month_of_max.dump("month_of_max_"+str(select_grid[0])+"deg_"+str(moving_avg_length[0])+"day.npy")
day_of_max.dump("day_of_max_"+str(select_grid[0])+"deg_"+str(moving_avg_length[0])+"day.npy")
hour_of_max.dump("hour_of_max_"+str(select_grid[0])+"deg_"+str(moving_avg_length[0])+"day.npy")

print("Arrays saved. Moving to create .png and .gif images.")
#Next, following the procedure from Mapes (2011), we rescale arrays to fit 0-255 byte range. This is to limit the dynamic revolution of the array data to 1/255 of the actual value. Let's set some parameters for rescaling based on our called grid:

#Bit packing for pcp (bp_multi):
if select_grid[0] == 0.25:
    bp_multi= (6/50) #Following Mapes (2011).
elif select_grid[0] == 1:
    bp_multi= 0.2
elif select_grid[0] == 2:
    bp_multi= 0.25
elif select_grid[0] == 4:
    bp_multi= 0.5
    
#Perform calculations to properly write the .png files.
pcp_max_rs = np.multiply(pcp_max, bp_multi)
pcp_max_rs = np.flipud(pcp_max_rs)
pcp_max_rs = bytescale(pcp_max_rs, low= pcp_max_rs.min(), high = pcp_max_rs.max())

year_of_max_rs= np.subtract(year_of_max, 1998)*14
year_of_max_rs = np.flipud(year_of_max_rs)
year_of_max_rs = bytescale(year_of_max_rs, low= year_of_max_rs.min(), high = year_of_max_rs.max())

month_of_max_rs = np.multiply(month_of_max, 20)
month_of_max_rs = np.flipud(month_of_max_rs)
month_of_max_rs = bytescale(month_of_max_rs, low= month_of_max_rs.min(), high= month_of_max_rs.max())

day_of_max_rs = np.multiply(day_of_max, 8)
day_of_max_rs = np.flipud(day_of_max_rs)
day_of_max_rs = bytescale(day_of_max_rs, low= day_of_max_rs.min(), high= day_of_max_rs.max())

hour_of_max_rs = np.multiply(hour_of_max, 12)
hour_of_max_rs = np.flipud(hour_of_max_rs)
hour_of_max_rs = bytescale(hour_of_max_rs, low= hour_of_max_rs.min(), high= hour_of_max_rs.max())

#Write the .png files packed with the data from the record events.
write_png((str(select_grid[0])+"deg_amountof"+str(moving_avg_length[0])+"dayrecord.png"), pcp_max_rs)
write_png((str(select_grid[0])+"deg_yearof"+str(moving_avg_length[0])+"dayrecord.png"), year_of_max_rs)
write_png((str(select_grid[0])+"deg_monthof"+str(moving_avg_length[0])+"dayrecord.png"), month_of_max_rs)
write_png((str(select_grid[0])+"deg_dayof"+str(moving_avg_length[0])+"dayrecord.png"), day_of_max_rs)
write_png((str(select_grid[0])+"deg_hourof"+str(moving_avg_length[0])+"dayrecord.png"), hour_of_max_rs)

#We need to resize the PNG files to match the dimensions on the website.

im1 = Image.open(str(select_grid[0])+"deg_amountof"+str(moving_avg_length[0])+"dayrecord.png")
im1 = im1.resize((1440,400))
im1.save(str(select_grid[0])+"deg_amountof"+str(moving_avg_length[0])+"dayrecord.png")

im2 = Image.open(str(select_grid[0])+"deg_yearof"+str(moving_avg_length[0])+"dayrecord.png")
im2 = im2.resize((1440,400))
im2.save(str(select_grid[0])+"deg_yearof"+str(moving_avg_length[0])+"dayrecord.png")

im3 = Image.open(str(select_grid[0])+"deg_monthof"+str(moving_avg_length[0])+"dayrecord.png")
im3 = im3.resize((1440,400))
im3.save(str(select_grid[0])+"deg_monthof"+str(moving_avg_length[0])+"dayrecord.png")

im4 = Image.open(str(select_grid[0])+"deg_dayof"+str(moving_avg_length[0])+"dayrecord.png")
im4 = im4.resize((1440,400))
im4.save(str(select_grid[0])+"deg_dayof"+str(moving_avg_length[0])+"dayrecord.png")

im5 = Image.open(str(select_grid[0])+"deg_hourof"+str(moving_avg_length[0])+"dayrecord.png")
im5 = im5.resize((1440,400))
im5.save(str(select_grid[0])+"deg_hourof"+str(moving_avg_length[0])+"dayrecord.png")

#Now onto the colored .gif images and matplotlib.

#Define the plot background here.
def plot_background(ax):
    ax.set_extent([lons.min(), lons.max(), lats.min(), lats.max()], crs= ccrs.PlateCarree())
    ax.coastlines('10m', edgecolor='black', linewidth=0.5)
    lon_formatter = LongitudeFormatter(zero_direction_label=True)
    lat_formatter = LatitudeFormatter()
    ax.xaxis.set_major_formatter(lon_formatter)
    ax.yaxis.set_major_formatter(lat_formatter)
    states_provinces = cfeature.NaturalEarthFeature(
        category='cultural',
        name='admin_1_states_provinces_lines',
        scale='10m',
        facecolor='none')
    ax.add_feature(states_provinces, edgecolor='gray')
    return ax

crs= ccrs.PlateCarree(central_longitude=0)

#Set the color table for the precipitation maxima.
bounds_pcp= np.linspace(0, np.around(pcp_max.max(), decimals=-1), num=15, endpoint=True)
norm_pcp = mpl.colors.BoundaryNorm(boundaries=bounds_pcp, ncolors=256)

#Now the color table for the year and month. These are more straightforward and we can just take the maximum:
bounds_yr= np.linspace(np.around(year_of_max.min(), decimals=0),np.around(year_of_max.max(), decimals=0), num= ((year_of_max.max()-year_of_max.min())+1), endpoint=True)
norm_yr= mpl.colors.BoundaryNorm(boundaries=bounds_yr, ncolors=256)

#And for the months:
bounds_mon= np.linspace(month_of_max.min(),month_of_max.max(), num= ((month_of_max.max())), endpoint=True)
norm_mon= mpl.colors.BoundaryNorm(boundaries=bounds_mon, ncolors=256)

cmap= 'gist_ncar'

#Create colored maps for display.
# create the figure and axes instances.
fig1 = plt.figure(figsize=(14.4, 4))
ax1= plt.subplot(projection=crs)
plot_background(ax1)

cf1= plt.pcolormesh(lons, lats, pcp_max, cmap=cmap, norm=norm_pcp, transform=ccrs.PlateCarree())
plt.savefig(str(select_grid[0])+"deg_amountof"+str(moving_avg_length[0])+"dayrecord_color.png", bbox_inches='tight', pad_inches=0.0)

fig2, ax2 = plt.subplots(figsize=(7,1), dpi=100)
cb2 = mpl.colorbar.ColorbarBase(ax2, cmap=cmap,
                                norm=norm_pcp,
                                orientation='horizontal')
cb2.set_label('Precipitation (mm)')
fig2.savefig(str(select_grid[0])+"deg_legend_amt_"+str(moving_avg_length[0])+"day.png", bbox_inches='tight', pad_inches=0.0)

fig3= plt.figure(figsize=(14.4, 4))
ax3= plt.subplot(projection=crs)
plot_background(ax3)

cf2= plt.pcolormesh(lons, lats, year_of_max, cmap=cmap, norm=norm_yr, transform=ccrs.PlateCarree())
plt.savefig(str(select_grid[0])+"deg_yearof"+str(moving_avg_length[0])+"dayrecord_color.png", bbox_inches='tight', pad_inches=0.0)

fig4, ax4 = plt.subplots(figsize=(7,1), dpi=100)
cb2 = mpl.colorbar.ColorbarBase(ax4, cmap=cmap,
                                norm=norm_yr,
                                orientation='horizontal')
cb2.set_label('Year')
fig4.savefig(str(select_grid[0])+"deg_legend_yearofrecord_"+str(moving_avg_length[0])+"day.png", bbox_inches='tight', pad_inches=0.0)

fig5= plt.figure(figsize=(14.4, 4))
ax5= plt.subplot(projection=crs)
plot_background(ax5)

cf3= plt.pcolormesh(lons, lats, month_of_max, cmap='gist_ncar', norm=norm_mon, transform=ccrs.PlateCarree())
plt.savefig(str(select_grid[0])+"deg_monthof"+str(moving_avg_length[0])+"dayrecord_color.png", bbox_inches='tight', pad_inches=0.0)

fig6, ax6 = plt.subplots(figsize=(7,1), dpi=100)
cb3 = mpl.colorbar.ColorbarBase(ax6, cmap=cmap,
                                norm=norm_mon,
                                orientation='horizontal')
cb3.set_label('Month')
fig6.savefig(str(select_grid[0])+"deg_legend_monthofrecord_"+str(moving_avg_length[0])+"day.png", bbox_inches='tight', pad_inches=0.0)

#Again, resize the colored .png images and convert them to .gif
im6 = Image.open(str(select_grid[0])+"deg_amountof"+str(moving_avg_length[0])+"dayrecord_color.png")
im6 = im6.resize((1440,400))
im6.save(str(select_grid[0])+"deg_amountof"+str(moving_avg_length[0])+"dayrecord_color.gif")

im7 = Image.open(str(select_grid[0])+"deg_yearof"+str(moving_avg_length[0])+"dayrecord_color.png")
im7 = im7.resize((1440,400))
im7.save(str(select_grid[0])+"deg_yearof"+str(moving_avg_length[0])+"dayrecord_color.gif")

im8 = Image.open(str(select_grid[0])+"deg_monthof"+str(moving_avg_length[0])+"dayrecord_color.png")
im8 = im8.resize((1440,400))
im8.save(str(select_grid[0])+"deg_monthof"+str(moving_avg_length[0])+"dayrecord_color.gif")

print(".png and .gif images created successfully!")
print("Done")
