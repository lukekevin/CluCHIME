'''
Author: Kevin Luke
'''
import sigpyproc
import numpy as np
import matplotlib.pyplot as plt
from sigpyproc.readers import FilReader
from astropy.time import Time as t, TimeDelta as td
from matplotlib import gridspec

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--filename', 
                        type=str, 
                        help='Dir where fil data is dumped')   
    parser.add_argument('--tarr', 
                        type=float, 
                        help='Arrival time of pulse in the data in s.')
    parser.add_argument('--dm', 
                        type=float, 
                        help='DM of pulse in the data.')
    parser.add_argument('--tdown', 
                          type=float, 
                          help='Time downsample to be used . Higher val means thicker pulse.')
    parser.add_argument('--freqsub', 
                        type=float, 
                        help='Subband factor to be used for the data. Higher val means less bands. Must be multiple of 2')
    parser.add_argument('--span', 
                        type=float, 
                        help='Range of data to be considered on either side of the pulse. In samples based on sampling time of data')
    parser.add_argument('--upfreq', 
                        type=float, 
                        help='Top frequency of the data') 
    parser.add_argument('--lowfreq', 
                        type=float, 
                        help='Lower frequency of the data') 
    
    args = parser.parse_args()
    
    filename=args.filename
    tarr=args.tarr
    dm=args.dm
    tdown=args.tdown
    freqsub=args.freqsub
    span=args.span
    upfreq=args.upfreq
    lowfreq=args.lowfreq
    
    #load the fil data and display the header info
    fil_path=filename
    fil_file=FilReader(fil_path)
    print(fil_file.header)

    tsamp = fil_file.header.tsamp
    time= tarr

    DM=dm
    tdownsamp=tdown
    subband=freqsub
    # Process the fil data with various utilities of SIGPYPROC
    tsample = int(time / tsamp)
    nsamples = span // tdownsamp * tdownsamp 
    data = fil_file.read_block(tsample - nsamples//2, 
                               nsamples)

    #zerodm the data
    data[data==0]=np.nan
    chan_means=np.nanmean(data,axis=0)
    data[:,:]=data[:,:] - chan_means[None,:]
    data[np.isnan(data)]=0

    data = data.dedisperse(DM)
    data = data.downsample(tdownsamp, subband)
    data = data.normalise()
    tseries = data.get_tim()  #/ np.sqrt(data.header.nchans)
    # Update sampling time and nsamples
    # dt is the smallest element on the time axis of data after DOWNSAMPLING BY SIGPYPROC
    dt = tdownsamp*tsamp
    # Update the number of samples on time axis  after DOWNSAMPLING BY SIGPYPROC
    nsamples = nsamples // tdownsamp

    # Calculate the absolute arrival time
    t_start = t(fil_file.header.tstart, format='mjd')
    t_arr_rel = td(time, format='sec')
    t_arr_abs = t_start + t_arr_rel
    print('Time of Arrival of pulse:\n', t_arr_abs.iso)

    # Initiate a figure and make 4 grids in it for various subplots
    fig = plt.figure(figsize=(20, 10),
                     dpi=100)
    spec = gridspec.GridSpec(ncols=2,
                             nrows=2,
                             height_ratios=[2, 4],
                             width_ratios=[2, 3])

    # Make ticks for the time and freq axes of waterfall
    xticklabels = np.round(np.linspace(-nsamples/2*dt,
                                       nsamples/2*dt, 51), 2)
    xticks = (xticklabels + xticklabels.max())/dt
    xlims = [xticks.max()/2 - 750/tdownsamp,
             xticks.max()/2 + 750/tdownsamp]

    freqs = np.linspace(upfreq, lowfreq, 9)
    yticks = (upfreq - freqs) / abs(data.header.foff)

    # Add timeseries
    ax1 = fig.add_subplot(spec[0])
    ax1.plot(tseries)
    ax1.set_xticks(xticks.astype(int))
    ax1.set_xlim(xlims)
    ax1.xaxis.set_visible(False)
    ax1.set_ylabel('SNR / bin ($\\sigma$ units)', fontsize=18)

    # Add the waterfall
    ax2 = fig.add_subplot(spec[2])
    imwf = ax2.imshow(data, aspect='auto',
                      vmax=5, vmin=-5)
    ax2.set_xticks(xticks.astype(int))
    ax2.set_xticklabels(np.round(xticklabels +time,1),
                        fontsize=12)
    ax2.set_xlim(xlims)
    ax2.set_xlabel("Time (s)", fontsize=18)
    ax2.set_yticks(yticks.astype(int))
    ax2.set_yticklabels(freqs, fontsize=12)
    ax2.set_ylabel("Frequency (MHz)",
                   fontsize=18)

    # Adjust the subplots and save the figure
    fig.subplots_adjust(.1, .1, .95, .95, 0, 0)
    outfname = "waterfaler.jpg"
    fig.savefig(outfname, dpi=300)
