import sigpyproc
import h5py
import numpy as np
import matplotlib.pyplot as plt
from sigpyproc.readers import FilReader
from sigpyproc.io.fileio import FileWriter
from sigpyproc.io import sigproc
from astropy.time import Time as t, TimeDelta as td
from matplotlib import gridspec
from sps_common.conversion import read_huff_msgpack
from sps_common.constants import TSAMP, FREQ_TOP, FREQ_BOTTOM


datfile='/DATA/chime-slow/raw/dat/2022/03/30/0059/1648601213_1648601232.dat'
channel_downsampling_factor=1
outsubfiles = read_huff_msgpack(datfile, channel_downsampling_factor)
ntime = outsubfiles[0]["ntime"]
spectra = outsubfiles[0]["spectra"].data.astype(np.float32)
mask = outsubfiles[0]["spectra"].mask


def create_filterbank_fileobj(tmpfname, 
                              nchans, nsamples, 
                              tsamp, tstart,
                              beam):
    chan_bw = np.abs(FREQ_TOP - FREQ_BOTTOM) / nchans
    header = dict(
        nsamples=nsamples,
        nchans=nchans,
        fch1=FREQ_TOP - chan_bw / 2.0,
        foff=-1.0 * chan_bw,
        nbeams=1,
        ibeam=int(beam),
        nifs=1,
        tsamp=tsamp,
        tstart=tstart,
        data_type=1,
        telescope_id=20,
        machine_id=20,
        nbits=32,
        barycentric=0,
        pulsarcentric=0,
        source_name="Stationary Beam",
        src_raj=0.0,
        src_dej=0.0,)

    fil_fileobj = FileWriter(tmpfname, 
                             mode="w", 
                             tsamp=tsamp, 
                             nchans=nchans, 
                             nbits=32)
    fil_fileobj.write(sigproc.encode_header(header))

    return fil_fileobj


fil_fileobj = create_filterbank_fileobj('dat_data.fil',
                                        *spectra.shape,
                                        0.00098304,
                                        1, 1)
fil_fileobj.cwrite(spectra)
fil_fileobj.close()




#load the fil data and display the header info
DM=30
tdownsamp=2
subband=4
tstart=6000
tend=6500

fil_path='dat_data.fil'
fil_file=FilReader(fil_path)
fil_file.header

tsamp = fil_file.header.tsamp
nsamples=fil_file.header.nsamples

# Process the fil data with various utilities of SIGPYPROC
data=fil_file.read_block(tstart,tend)


data[data==0]=np.nan
chan_means=np.nanmean(data,axis=0)
data[:,:]=data[:,:] - chan_means[None,:]
data[np.isnan(data)]=0


data = data.dedisperse(DM)
data = data.downsample(tdownsamp, subband)
data = data.normalise()
tseries = data.get_tim()/np.sqrt(data.header.nchans)

fig = plt.figure(figsize=(10, 5), dpi=100)
spec = gridspec.GridSpec(ncols=2, nrows=2, 
                         height_ratios=[2, 4],
                         width_ratios=[2, 3])

freqs = np.linspace(800, 400., 9)
yticks = (800 - freqs) / abs(data.header.foff)

# Add timeseries
ax1 = fig.add_subplot(spec[0])
ax1.plot(tseries)
ax1.xaxis.set_visible(False)
ax1.set_ylabel('SNR / bin ($\\sigma$ units)', fontsize=18)


# Add the waterfall
ax2 = fig.add_subplot(spec[2])
imwf = ax2.imshow(data, aspect='auto', vmax=5, vmin=-5)
ax2.set_xlabel("Timesamples", fontsize=18)
ax2.set_ylabel("Frequencybands", fontsize=18)


# Adjust the subplots and save the figure
fig.subplots_adjust(.1, .1, .95, .95, 0, 0)
outfname = "pulsar_waterfaler.jpg"
fig.savefig(outfname, dpi=300)
