"""
Creating a filterbank file with a fake frb in it. 
The code is an improved NON hardcoded version of Kendricks Smith's original
example code for generating a fake frb with additional utility of pumping the created frb
inside a GBT breakthrough file format Filterbank file. 
The code is found at 
https://kmsmith137.github.io/simpulse_docs/single_pulse.html
""""


import simpulse
import matplotlib.pyplot as plt
import numpy as np
import argparse
from sigpyproc.io.fileio import FileWriter
from sigpyproc.io import sigproc

def create_filterbank_fileobj(tmpfname, nchans, 
                              nsamples, tsamp, 
                              tstart, beam):
        """
        filterbank object header
        """
        chan_bw = np.abs(freq_top - freq_low) / nchans
        header = dict(
            nsamples=nsamples,
            nchans=nchans,
            fch1=800 - chan_bw / 2.0,
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
            src_dej=0.0)

        fil_fileobj = FileWriter(tmpfname,
                                 mode="w",
                                 tsamp=tsamp,
                                 nchans=nchans,
                                 nbits=32)
        fil_fileobj.write(sigproc.encode_header(header))

        return fil_fileobj



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--out_dir', 
                        type=str, 
                        help='Dir where fake data is dumped')   
    parser.add_argument('--mu', 
                        type=float,
                        help='Mu of the normal distribution data.',
                        default=1.0)
    parser.add_argument('--sigma', 
                        type=float, 
                        help='Sigma of the normal distribution data.',
                        default=1.0)
    
    parser.add_argument('--nsecs', 
                        type=float, 
                        help='No of secs in data.',
                        default=10.0)
    
    parser.add_argument('--nchannels', 
                        type=int, 
                        help='No of channels in data.',
                        default=1000)
    
    parser.add_argument('--sampling_time', 
                        type=float, 
                        help='Sampling of time in the data in s.',
                        default=0.001)
    parser.add_argument('--snr', 
                        type=float, 
                        help='SNR of pulse in the data.',
                        default=100.0)  
    parser.add_argument('--dm', 
                        type=float, 
                        help='DM of pulse in the data.',
                        default=100.0)  
    parser.add_argument('--freq_top', 
                        type=float, 
                        help='Top frequency of the filterbank data.',
                        default=800.0)  
    parser.add_argument('--freq_low', 
                        type=float, 
                        help='Bottom frequency of the filterbank data.',
                        default=400.0)  
    parser.add_argument('--width', 
                        type=float, 
                        help='Width of pulse in seconds.',
                        default=0.01)  
    parser.add_argument('--tarr', 
                        type=float, 
                        help='Undispersed arrival time of the pulse in seconds.',
                        default=1.0)  
    parser.add_argument('--rms', 
                        type=float, 
                        help='RMS of data.',
                        default=10.0)  
    args = parser.parse_args()
    
    mu = args.mu
    sigma = args.sigma
    nchans = args.nchannels
    t1 = args.nsecs
    tsamp = args.sampling_time
    out_dir = args.out_dir    
    target_snr = args.snr
    dm=args.dm
    width=args.width
    freq_top = args.freq_top
    freq_low = args.freq_low
    sample_rms= args.rms
    nsamples = int(t1*1/tsamp)
    t0=0.0  #secs, start of the file
    
    #Generate a guassian distribution data of size nchans and nsamples
    data=np.random.normal(mu,sigma, size=(nchans,nsamples))

    #Simpulse pulse maker for the single pul
    p = simpulse.single_pulse(nt = nsamples,             
                              nfreq = nchans,           
                              freq_lo_MHz = freq_low,   
                              freq_hi_MHz = freq_top,  
                              dm = dm,           
                              sm = 0.0,              
                              intrinsic_width = width, 
                              fluence = 1.0,        
                              spectral_index = 0.0,  
                              undispersed_arrival_time = 1.0)

    initial_snr = p.get_signal_to_noise(sample_dt=(t1-t0)/nsamples, 
                                        sample_rms=sample_rms)
    p.fluence *= (target_snr / initial_snr)
    
    #Add the simpulse generated frb to the gaussian data
    p.add_to_timestream(data, t0, t1)
    
    #Name of the filterbank filewhich will be generated
    name="""frbpulse_{0:f}_nchans_{1:f}_nsamples_{2:f}_sampling.fil""".format(nchans, 
                                                                               nsamples ,
                                                                               tsamp)
    #Filterbank header maker
    fil_fileobj = create_filterbank_fileobj(out_dir+"/"+name, 
                                            *data.shape, 
                                            tsamp,
                                            1, 1)
    
    print("""Inject a fake frb and creating a filterbank file with nchannels {0:f} ,
    nsamples {1:f} ,sampling,{2:f}""".format(nchans, 
                                             nsamples,
                                             tsamp))
    
    #Plot the pulse and save it
    plt.imshow(data, interpolation='none', 
               origin='lower',
               extent=(t0, t1, p.freq_lo_MHz, p.freq_hi_MHz),
               aspect='auto')
    plt.savefig(out_dir+'/'+'frb.jpg')
    
    #Write the data to filterbank file with the given header
    fil_fileobj.cwrite(data)
    fil_fileobj.close()
