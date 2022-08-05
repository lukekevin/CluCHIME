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
import os
import argparse

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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dat_file',
                        type=str, 
                        help='Path where dat file is stored')
    parser.add_argument('--out_dir_path', 
                        type=str, 
                        help='Path where the filterbank data will be dumped')
    args = parser.parse_args()
    
    dat_file=args.dat_file
    out_dir_path=args.out_dir_path
     
    out_dir = out_dir_path + 'dat_to_filterbank_file' + "/"
    os.makedirs(out_dir, exist_ok=True)

    datfile=dat_file
    channel_downsampling_factor=1
    outsubfiles = read_huff_msgpack(datfile, channel_downsampling_factor)
    ntime = outsubfiles[0]["ntime"]
    spectra = outsubfiles[0]["spectra"].data.astype(np.float32)
    mask = outsubfiles[0]["spectra"].mask

    fil_fileobj = create_filterbank_fileobj(out_dir+'/'+'dat_data.fil',
                                            *spectra.shape,
                                            0.00098304,
                                            1, 1)
    fil_fileobj.cwrite(spectra)
    fil_fileobj.close()
