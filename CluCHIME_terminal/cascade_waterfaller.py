#!/usr/bin/env python
# coding: utf-8
'''
Author: Kevin Luke 
CluCHIME.py was made for MSc thesis project at TIFR, Mumbai
Date created: 23 Nov 2021
Date last modified: 10 June 2022
'''

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

import numpy as np
from matplotlib import gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
from numpy import savez_compressed 

import datetime, time

from frb_common import common_utils
from iautils import cascade, spectra, spectra_utils
from iautils.conversion import chime_intensity
from numpy import inf
import glob 

def Waterfaller(FinalData,
                TIMESERIES,
                start, 
                end,
                out_dir) :
    """
   Generate a waterfall plot for a range of timestamps, plot the timeseries for the
   range of timestamps and save them.

   Parameters
   -----------
   FinalData : str
          String of the path where the final data from the previous step is stored.
   TIMESERIES : str
          String of the path where the timeseries from previous step is stored.
   start : int
          Approximate starting timestamp for the time range on timeseries obtained from previous function.
   end : int
          Approximate ending timestamp for the time range on timeseries obtained from previous function.

   Returns
   -----------
   Wterfall_Timeseries.jpg : jpg
          Waterfaller plot and timeseries of the event.

   """

    FinalData=np.load(FinalData)
    FinalData=FinalData['arr_0']

    TIMESERIES=np.load(TIMESERIES)
    TIMESERIES=TIMESERIES['arr_0']

    #MANUAL WATERFALLING
    time_series=TIMESERIES[self.start:self.end]/np.sqrt(FinalData.shape[0])
    SNR=spectra.smoothed_peak_snr(timeseries=time_series)
    print('smoothed peaked SNR:',SNR[0])
    get_ipython().run_line_magic('matplotlib', 'inline')


    fig = plt.figure(figsize=(10, 10), dpi=50)
    spec = gridspec.GridSpec(ncols=1, 
                             nrows=2, 
                             height_ratios=[2, 4])

    ax1 = fig.add_subplot(spec[0])
    ax1.plot(time_series)
    ax1.set_ylabel('SNR / bin ($\\sigma$ units)', fontsize=18)

    fbins=np.round(np.linspace(0,FinalData.shape[0],10))
    ax2 = fig.add_subplot(spec[1])
    imwf = ax2.imshow(np.flip((FinalData[:,self.start:self.end]),axis=0),
                      aspect="auto",
                      vmax=10, vmin=-10)
    ax2.set_yticks(fbins)
    ax2.set_yticklabels(np.round(np.flip(np.linspace(400,800,10))))
    ax2.set_ylabel("Frequency (MHz)", fontsize=18)
    tbins=np.round(np.linspace(0,(FinalData[:,self.start:self.end]).shape[1],10),1)
    ax2.set_xticks(tbins)
    ax2.set_xticklabels(np.round(np.linspace(self.start/1000,self.end/1000,10),2))
    ax2.set_xlabel("Time (s)", fontsize=18)

   # Adjust the colorbar on this axes alongside tables

    divider = make_axes_locatable(ax2)
    cax = divider.append_axes("right", 
                              size="1%", 
                              pad=0.001)
    plt.colorbar(imwf, cax=cax)
    fig.subplots_adjust(.1, .1, 
                        .95, .95, 
                        0, 0)
    fig.savefig(out_dir+'Waterfall_Timeseries.jpg', dpi=300)

    spectra.find_burst((TIMESERIES[self.start:self.end]), 
                              width_factor=10, 
                              min_width=1, 
                              max_width=10, 
                              plot=True)
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--FINALDATA',
                        type=str, 
                        help='Data path where the subbanded, dedispersed array is stored.')
    parser.add_argument('--TIME_SERIES',
                        type=str, 
                        help='Data path where the dedispersed timeseries is stored.')
    parser.add_argument('--START',
                        type=int, 
                        help='Aproximate start timestamp to be selected for waterfalling.')
    parser.add_argument('--END',
                        type=int, 
                        help='Aproximate end timestamp to be selected for waterfalling.')
    parser.add_argument('--out_dir_path', 
                        type=str, 
                        help='Path where the plots will be dumped')
    args = parser.parse_args()    
    
    args.FINALDATA=FINALDATA
    args.TIME_SERIES=TIME_SERIES
    args.START=START
    args.END=END
    
    out_dir = out_dir_path + 'waterfaller' + "/"
    os.makedirs(out_dir, exist_ok=True)
    
    Waterfaller(FINALDATA,
                TIME_SERIES,
                START, 
                END)