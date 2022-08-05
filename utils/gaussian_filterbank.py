import numpy as np
import argparse

from sigpyproc.io.fileio import FileWriter
from sigpyproc.io import sigproc


def create_filterbank_fileobj(tmpfname, nchans, 
                              nsamples, tsamp, 
                              tstart, beam):
  
        """
        Collect all the individual MSGPACK INTENSITY  datapackets of CHIME/FRB L1 NODE 
        from the specified path and convert them to numpy array and then save them 
        as .npz files.
        Parameters
        ----------
        tmpfname : str
            Name of the filterbank to be made.
        nchans   : int
            No of channels in the msgpack file.
        nsamples : int
            No of time samples in the msgpack file.
        tsamp    : float
            Sampling time of the CHIME/FRB msgpack data and HENCE the filterbank file.
        tstart   : float
            Start time in the filterbank file.
        beam     : int
            Beam number to be written in the filterbank file.
        Returns
        -------
        filterbank object
        """
    
    chan_bw = np.abs(800 - 400) / nchans
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
                        type=int, 
                        help='Mu of the normal distribution data.')
    parser.add_argument('--sigma', 
                        type=float, 
                        help='Sigma of the normal distribution data.')
    parser.add_argument('--nchannels', 
                        type=int, 
                        help='No of frequency channels in data.')
    parser.add_argument('--nsamples', 
                        type=int, 
                        help='Timesamples in the data.')
    parser.add_argument('--sampling_time', 
                        type=float, 
                        help='Sampling of time in the data in s.')
    
    args = parser.parse_args()
    
    mu = args.mu
    sigma = args.sigma
    nchannels = args.nchannels
    nsamples = args.nsamples
    sampling_time = args.sampling_time
    out_dir = args.out_dir
    
    print("""Fake data generator in filterbank format, telescope id is set by default to CHIME""")
    
    #generate the fake data array
    fake_data=np.random.normal(mu, sigma,size=(nchannels, nsamples))

    #Name of the filterbank filewhich will be generated
    name="""fake_sky_data_{0:f}_nchans_{1:f}_nsamples_{2:f}_sampling.fil""".format(nchannels, 
                                                                               nsamples ,
                                                                               sampling_time)
    
    #Creating the filterbank object
    fil_fileobj = create_filterbank_fileobj(out_dir+"/"+name,
                                            *fake_data.shape,
                                            sampling_time,
                                            1, 1)
    
    print("""Creating fake data in filterbank format with nchannels {0:f} ,
    nsamples {1:f} ,sampling,{2:f}""".format(nchannels, 
                                             nsamples,
                                             sampling_time))
    
    #write the data to the filterbank object and dump it in the current directory
    fil_fileobj.cwrite(fake_data)
    fil_fileobj.close()
