import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

import numpy as np
from matplotlib import gridspec

import hdbscan
from numpy import savez_compressed 

import datetime, time
from frb_common import common_utils
from iautils import cascade, spectra, spectra_utils
from iautils.conversion import chime_intensity
from numpy import inf

def Iautils(INTnewnorm_whole, 
            INT_un, 
            Weight, 
            fpga0,
            fpgan,
            dm,
            beam,
            out_dir):
    """

    The regular workflow on in this code block is as follows:
    1) Declare various constants and input parameters to be passed to cascade object in
    step 2.
    2) Make a cascade object containing data/ spectrum with cascade.py .
    3) Dedisperse the data from cascade object.
    4) Subband the data from  cascade object.
    5) Save the processed data and the timeseries.

    Parameters
    -----------
    INTnewnorm_whole : str
         String of path to the dir where the cleaned and normalised data obtained from previous step is stored.
    INT_un : str
         String of path to the dir where the unclean and unnormalised data is stored.
    Weight : str
         String of path of weights array for all 3 beams.
    fpga0 : str
         String of path of fpga0 values for all 3 beams.
    fpgan : str
         String of path of fpgan values for all 3 beams.
    dm : float
         Float value of the DM for dedispersion.
    beam : str
         Beam number to be processed.

    Returns 
    -----------
    Timeseries plot : Plot in matplotlib inline
         Use the range in the timeseries where the pulse is present for the Waterfaller function.
    FinalData : npz file
         Final data array block after dedispersion, subbanding is done.
    TIMESERIES : npz file
         Timeseries array obtained for the final data block.

    """

    #IAUTILS
    
    INT_un=np.load(INT_un)
    INT_un=INT_un['arr_0']

    INTnewnorm_whole=np.load(INTnewnorm_whole)
    INTnewnorm_whole=INTnewnorm_whole['arr_0']

    Weight=np.load(Weight)
    Weight=Weight['arr_0']

    fpga0=np.load(fpga0)
    fpga0=fpga0['arr_0']

    fpgan=np.load(fpgan)
    fpgan=fpgan['arr_0']

    INT_un1st=INT_un[0,:,:]
    INT_un2nd=INT_un[1,:,:]
    INT_un3rd=INT_un[2,:,:]

    INTnewnorm_whole1st=INTnewnorm_whole[0,:,:]
    INTnewnorm_whole2nd=INTnewnorm_whole[1,:,:]
    INTnewnorm_whole3rd=INTnewnorm_whole[2,:,:]

    Weight1=Weight[0,:,:]
    Weight2=Weight[1,:,:]
    Weight3=Weight[2,:,:]

    fpga01st=fpga0[0]
    fpga02nd=fpga0[1]
    fpga03rd=fpga0[2]

    fpgan1st=fpgan[0]
    fpgan2nd=fpgan[1]
    fpgan3rd=fpgan[2]

    #making copies of arrays
    import copy 
    INTnewnorm_whole1st=copy.copy(INTnewnorm_whole1st)
    INT_un1st=copy.copy(INT_un1st)

    INTnewnorm_whole2nd=copy.copy(INTnewnorm_whole2nd)
    INT_un2nd=copy.copy(INT_un2nd)

    INTnewnorm_whole3rd=copy.copy(INTnewnorm_whole3rd)
    INT_un3rd=copy.copy(INT_un3rd)


    #meanSTD removal
    '''
    INTcascade=INTnewnorm_whole2nd

    INTmeanstd=[]
    for i in range(len(INTcascade)):
        MEAN=np.nanmean(INTcascade[i,:])
        INTMEANREMOVED=INTcascade[i,:]-MEAN
        INTSTD=np.nanstd(INTcascade[i,:])
        INTMEANSTD=INTMEANREMOVED/INTSTD
        INTmeanstd.append(INTMEANSTD)
    INTmeanstd=np.array(INTmeanstd)

    '''

    event_time = datetime.datetime.utcnow()
    fpga_time=481426529279
    time_range = 0.1
    dm_range = 0.5
    beam = beam

    if beam=='1_UNCLEAN':
        INTEN=INT_un1st
        WEIGHT=Weight1
        FPGA0=fpga01st
        FPGAN=fpgan1st

    elif beam=='1_CLEAN':
        INTEN=INTnewnorm_whole1st
        WEIGHT=Weight1
        FPGA0=fpga01st
        FPGAN=fpgan1st

    elif beam=='2_UNCLEAN':
        INTEN=INT_un2nd
        WEIGHT=Weight2
        FPGA0=fpga02nd
        FPGAN=fpgan2nd

    elif beam=='2_CLEAN':
        INTEN=INTnewnorm_whole2nd
        WEIGHT=Weight2
        FPGA0=fpga02nd
        FPGAN=fpgan2nd

    elif beam=='3_UNCLEAN':
        INTEN=INT_un3rd
        WEIGHT=Weight3
        FPGA0=fpga03rd
        FPGAN=fpgan3rd

    elif beam=='3_CLEAN':
        INTEN=INTnewnorm_whole3rd
        WEIGHT=Weight3
        FPGA0=fpga03rd
        FPGAN=fpgan3rd 

    #Cascade object from cascade.py     
    event = cascade.Cascade(intensities=[INTEN], 
                                weights=[WEIGHT], 
                                beam_nos=[beam],
                                fbottom=common_utils.freq_bottom_mhz,
                                df=common_utils.channel_bandwidth_mhz, 
                                tstart=0.0,
                                dt=common_utils.sampling_time_ms/1e3*1, 
                                fpga0s=[FPGA0],
                                fpgaNs=[FPGAN], 
                                dm= dm,
                                event_time=event_time,
                                fpga_time=int(fpga_time), 
                                time_range=float(time_range),
                                dm_range=float(dm_range),

                                use_rfi_masks=False,
                               )

   #DEDISPERSE
    event.dm = dm #558.4408569335938 #558.46   #558.4408569335938


   #SUBBANDING
    event.process_cascade(dm=dm, 
                          nsub=64, 
                          downsample_factor=1,
                          dedisperse=True, 
                          subband=True,
                          downsample=True, 
                          mask=False,
                          zerodm=False, 
                          scaleindep=True,
                          )

    #making timeseries
    TIMESERIES=((event.beams[0].generate_timeseries())[0])
    get_ipython().run_line_magic('matplotlib', 'inline')
    plt.plot(TIMESERIES)
    plt.savefig(out_dir+'TIMESERIES_PLOT.jpg')
    plt.show()   

    FinalData=event.beams[0].intensity
    savez_compressed(out_dir+'FinalData.npz',FinalData)
    savez_compressed(out_dir+'TIMESERIES.npz',TIMESERIES)
    
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--CLEANED_DENORM_ARRAY',
                        type=str, 
                        help='Path where cleaned denormalised array is stored.')
    parser.add_argument('--UNCLEANED_DENORM_ARRAY',
                        type=str, 
                        help='Path where uncleaned denormalised array is stored.')
    parser.add_argument('--FPGA0',
                        type=str, 
                        help='Path where FPGA0 file  is stored.')
    parser.add_argument('--FPGAN',
                        type=str, 
                        help='Path where FPGAN file  is stored.')
    parser.add_argument('--BEAM',
                        type=str, 
                        help=('BEAM NO to use and Choose from 1_UNCLEAN,1_CLEAN,2_UNCLEAN,2_CLEAN,3_UNCLEAN,3_CLEAN')
    parser.add_argument('--FPGAN',
                        type=str, 
                        help='Path where FPGAN file  is stored.')
    parser.add_argument('--WEIGHT',
                        type=str, 
                        help='Path where WEIGHT file  is stored.')           
    parser.add_argument('--DM',
                        type=float, 
                        help='DM value to use')
    parser.add_argument('--out_dir_path', 
                        type=str, 
                        help='Path where the processed data will be dumped')
    args = parser.parse_args()
                        
    args.CLEANED_DENORM_ARRAY=CLEANED_DENORM_ARRAY
    args.UNCLEANED_DENORM_ARRAY=UNCLEANED_DENORM_ARRAY
    args.FPGA0=FPGA0                    
    args.FPGAN=FPGAN
    args.BEAM=BEAM
    args.WEIGHT=WEIGHT
    args.DM=DM
    args.out_dir_path          
                        
    out_dir = out_dir_path + 'cascade_subbanding' + "/"
    os.makedirs(out_dir, exist_ok=True)                    
     
    Iautils(CLEANED_DENORM_ARRAY, 
            UNCLEANED_DENORM_ARRAY, 
            WEIGHT, 
            FPGA0,
            FPGAN,
            DM,
            BEAM,
            out_dir)
         

           
