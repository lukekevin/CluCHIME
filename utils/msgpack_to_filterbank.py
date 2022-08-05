import argparse
import numpy as np
from numpy import savez_compressed 
from iautils.conversion import chime_intensity
import glob 
from sigpyproc.io.fileio import FileWriter
from sigpyproc.io import sigproc

def collate(path1, 
            out_dir):
    """
    Collect all the individual MSGPACK INTENSITY  datapackets of CHIME/FRB L1 NODE 
    from the specified path and convert them to numpy array and then save them 
    as .npz files.
    Parameters
    ----------
    path1 : str
        Path to the directory where data packets for beam data is stored.
    out_dir : str
        Path where to save the results.
    Returns
    -------
    INT_un : npz file
        Unnormalised array having data from all 3 beams.  
    Weight : npz file
        Weights array for the beam.
    fpgan : npz file
        Fpgan values for the beam.
    fpga0 : npz file
        Fpga0 values for the beam.
    rfi_mask : npz file
        RFI mask for the beam.
    frame_nano : npz file
        Frame nano values for the beam.
    bins : npz file
        Bins values for the beam.
    """

    filelist1=glob.glob(path1)
    filelist1.sort()
    (I1,Weight1,
     bins1,fpga01st,
     fpgan1st,frame_nano1,
     rfi_mask1)=chime_intensity.unpack_datafiles(filelist1)

    #metadata
    Weight=np.array([Weight1])
    savez_compressed(out_dir+'Weight.npz',Weight)
    fpgan=np.array([fpgan1st])
    savez_compressed(out_dir+'fpgan.npz',fpgan)
    fpga0=np.array([fpga01st])
    savez_compressed(out_dir'+'fpga0.npz',fpga0)
    rfi_mask=np.array([rfi_mask1])
    savez_compressed(out_dir+'rfi_mask.npz',rfi_mask)
    frame_nano=np.array([frame_nano1])
    savez_compressed(out_dir+'frame_nano.npz',frame_nano)
    bins=np.array([bins1])
    savez_compressed(out_dir+'bins.npz',bins)

    #actual data array
    INT_un=np.array([I1])  #unclean unnormalised mega array
    savez_compressed(out_dir+'INT_un.npz',INT_un)
    return INT_un
    
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
    parser.add_argument('--pathname',
                        type=str, 
                        help='Data path for glob process where .msgpack for beam 1 is stored')
    parser.add_argument('--out_dir_path', 
                        type=str, 
                        help='Path where the processed data will be dumped')
    args = parser.parse_args()
    
    args.pathname=pathname
    args.out_dir_path=out_dir_path
     
    out_dir = out_dir_path + 'msgpack_to_npz_converted_file' + "/"
    os.makedirs(out_dir, exist_ok=True)
    INT_un=collate(pathname,out_dir)
    
    #Name of the filterbank filewhich will be generated
    name="""fil_data.fil"""
    #Creating the filterbank object
    fil_fileobj = create_filterbank_fileobj(out_dir+"/"+name,
                                            *INT_un.shape,
                                             0.00098304,
                                             1, 1)
    
    fil_fileobj.cwrite(fake_data)
    fil_fileobj.close()
