import sigpyproc
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import transforms, pyplot
from sigpyproc.readers import FilReader
from sigpyproc.io import sigproc
from astropy.time import Time as t, TimeDelta as td
from matplotlib import gridspec
import argparse

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('fil_path', 
                        type=str,
                        help='Path where filterbank data is stored')
    parser.add_argument('--subband', 
                        type=int, 
                        default=8,
                        help='No of subband to be made.')
    parser.add_argument('--time', 
                        type=float, 
                        default=800,
                        help='Time of arrival of the undedispersed pulse')
    parser.add_argument('--DM', 
                        type=float,
                        default=50,
                        help='DM to which the data will be dedispersed')
    parser.add_argument('--tdownsamp', 
                        type=int, 
                        default=1,
                        help='Time downsampling')
    parser.add_argument('--sweep', 
                        type=int, 
                        default=8192,
                        help='Range of timesamples on either sides of the pulse')
    parser.add_argument('--freqtop', 
                        type=float, 
                        default=800.,
                        help='Top frequency of the data')
    parser.add_argument('--freqbottom', 
                        type=float, 
                        default=400.,
                        help='Bottom frequency')
    parser.add_argument('out_path',
                        type=str,
                        help='Out path where the plots will be dumped')

    args = parser.parse_args()
    
    fil_path=args.fil_path
    time=args.time
    DM=args.DM
    tdownsamp=args.tdownsamp
    subband=args.subband
    out_path=args.out_path
    sweep=args.sweep
    freqtop=args.freqtop
    freqbottom=args.freqbottom
    
    fil_file=FilReader(fil_path)

    tsamp = fil_file.header.tsamp
    # Process the fil data with various utilities of SIGPYPROC
    tsample = int(time / tsamp)
    nsamples = sweep // tdownsamp * tdownsamp # always read +/- 4096 samples as max disp delay at DM 3k will be within these
    data = fil_file.read_block(tsample - nsamples//2, nsamples)
    data[data==0]=np.nan
    chan_means=np.nanmean(data,axis=0)
    data[:,:]=data[:,:] - chan_means[None,:]
    data[np.isnan(data)]=0
    data = data.dedisperse(DM)
    data = data.downsample(tdownsamp, subband)
    data = data.normalise()
    tseries = data.get_tim()
    bpass=data.get_bandpass()
    dt = tdownsamp*tsamp
    # Update the number of samples on time axis  after DOWNSAMPLING BY SIGPYPROC
    nsamples = nsamples // tdownsamp

    # Calculate the absolute arrival time
    t_start = t(fil_file.header.tstart, format='mjd')
    t_arr_rel = td(time, format='sec')
    t_arr_abs = t_start + t_arr_rel
    print('Time of Arrival of pulse:\n', t_arr_abs.iso)

    # Make ticks for the time and freq axes of waterfall
    #TO ZOOM IN OR OUT PLAY with 300 number in xlims and 100 number in xticklabels
    #If you increase the number in xlims then it will zoom out and then pls reduce the 
    #number in xtickslablels and vice versa
    xticklabels = np.round(np.linspace(-nsamples/2*dt, nsamples/2*dt, 250), 2)
    xticks = (xticklabels + xticklabels.max())/dt
    xlims = [xticks.max()/2 - 80/tdownsamp, xticks.max()/2 + 80/tdownsamp]
    #xlims=[xticks[0],xticks.max()]

    # Initiate a figure and make 4 grids in it for various subplots
    fig = plt.figure(figsize=(10, 10), dpi=100)
    spec = gridspec.GridSpec(ncols=2, nrows=2, height_ratios=[4, 8], width_ratios=[5, 1])


    freqs = np.linspace(freqtop, freqbottom, 9)
    yticks = (freqtop - freqs) / abs(data.header.foff)
    ylims=[data.shape[0],0]
    # Add timeseries
    ax1 = fig.add_subplot(spec[0])
    ax1.plot(tseries)
    ax1.set_xticks(xticks.astype(int))
    ax1.set_xlim(xlims)
    ax1.xaxis.set_visible(False)
    ax1.set_ylabel('SNR / bin ($\\sigma$ units)', fontsize=18)

    # Add the waterfall
    ax2 = fig.add_subplot(spec[2])
    imwf = ax2.imshow(data, aspect='auto', vmax=5, vmin=-5)
    ax2.set_xticks(xticks.astype(int))
    ax2.set_xticklabels(np.round(xticklabels +time,1), fontsize=12)
    ax2.set_xlim(xlims)
    ax2.set_xlabel("Time (s)", fontsize=18)
    ax2.set_yticks(yticks.astype(int))
    ax2.set_yticklabels(freqs, fontsize=12)
    ax2.set_ylabel("Frequency (MHz)", fontsize=18)

    #add bandpass
    ax3 = fig.add_subplot(spec[3])
    base = pyplot.gca().transData
    rot = transforms.Affine2D().rotate_deg(90)
    ax3.set_yticks(yticks.astype(int))
    ax3.set_yticklabels(freqs, fontsize=12)
    ax3.set_ylim(ylims)
    ax3.yaxis.set_visible(False)
    ax3.plot(-bpass,transform=rot+base)

    # Adjust the subplots and save the figure
    fig.subplots_adjust(.1, .1, .95, .95, 0, 0)
    outfname = out_path+"/"+"waterfaller.jpg"
    fig.savefig(outfname, dpi=300)
