#!/usr/bin/env python
# coding: utf-8
'''
Author: Kevin Luke 
CluCHIME.py was made for MSc thesis project at TIFR, Mumbai
Date created: 23 Nov 2021
Date last modified: 25 May 2022
'''


import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

import numpy as np
from matplotlib import gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable

import time
from numpy import asarray
from numpy import savetxt 

import hdbscan
import seaborn as sns
from numpy import save
from numpy import savez_compressed 

import datetime, time
from frb_common import common_utils
from iautils import cascade, spectra, spectra_utils
from iautils.conversion import chime_intensity
from numpy import inf
import glob 


class CluCHIME:
    ''' 
    For running CluCHIME library successfully you will
    need NUMPY, SCIPY, Scikit-Learn, IAUTILS and frb-common utils preinstalled.
    The later 2 libraries are closed CHIME/FRB libraries from GITHUB. The code 
    is slow and not efficiet in terms of memory usage and needs to updated 
    accordingly in future. The average run times for CLUSTERING ANALYSIS on 
    full data is ~6 hrs on  Dr. Shriharsh Tendulkar's workstation PRITHVI 
    at TIFR which has 128 GB ram, 14 TB of disk space. On regular PC's
    the run times will be much higher.
    '''
    
    def __init__(self,function):
        ''' Initialise the CluCHIME class'''
        
        self.function=function #input any string here 
        
        
        
    def Prepdata(self,
                 pathname1 , 
                 pathname2,
                 pathname3):
        """
         
        Preparing the data for clustering by collecting the individual MSGPACK data packets,
        collating them and then normalising them.
        
        Parameters
        ----------
        pathname1 : str
            Path to the directory where data packets for 1st beam is stored.
        pathname2 : str
            Path to the directory where data packets for 2nd beam is stored. 
        pathname3 : str
            Path to the directory where data packets for 3rd beam is stored.
       
        Returns
        -------
        INT_combined1 : npz file
            Normalisation array for all 3 beams.
        INT_un : npz file
            Unnormalised array having data from all 3 beams.  
        INT_n : npz file
            Normalised array having data from all 3 beams. 
        Weight : npz file
            Weights array for all 3 beams.
        fpgan : npz file
            Fpgan values for all 3 beams.
        fpga0 : npz file
            Fpga0 values for all 3 beams.
        rfi_mask : npz file
            RFI mask for all 3 beams.
        frame_nano : npz file
            Frame nano values for all 3 beams.
        bins : npz file
            Bins values for all 3 beams.
        
        """
        Intro='Author: Kevin Luke \nCluCHIME.py was made for MSc thesis project at TIFR, Mumbai \nDate created: 23 Nov 2021 \nDate last modified: Jan 20 2022'
        print(Intro)
        print('Collect data from dirs and do basic processing on them like normalisation')
        self.pathname1=pathname1
        self.pathname2=pathname2
        self.pathname3=pathname3
        
        #Collecting the data from designated path
        filelist1=glob.glob(self.pathname1)
        filelist1.sort()
        (I1,
         Weight1,
         bins1,
         fpga01st,
         fpgan1st,
         frame_nano1,
         rfi_mask1)=chime_intensity.unpack_datafiles(filelist1)


        filelist2=glob.glob(self.pathname2)
        filelist2.sort()
        (I2,
         Weight2,
         bins2,
         fpga02nd,
         fpgan2nd,
         frame_nano2,
         rfi_mask2)=chime_intensity.unpack_datafiles(filelist2)


        filelist3=glob.glob(self.pathname3)
        filelist3.sort()
        (I3,
         Weight3,
         bins3,
         fpga03rd,
         fpgan3rd,
         frame_nano3,
         rfi_mask3)=chime_intensity.unpack_datafiles(filelist3)
         
        #Savingthe metadata from .msgpack data    
        #metadata
        Weight=np.array([Weight1,Weight2,Weight3])
        savez_compressed('Weight.npz',Weight)

        fpgan=np.array([fpgan1st,fpgan2nd,fpgan3rd])
        savez_compressed('fpgan.npz',fpgan)

        fpga0=np.array([fpga01st,fpga02nd,fpga03rd])
        savez_compressed('fpga0.npz',fpga0)

        rfi_mask=np.array([rfi_mask1,rfi_mask2,rfi_mask3,])
        savez_compressed('rfi_mask.npz',rfi_mask)

        frame_nano=np.array([frame_nano1,frame_nano2,frame_nano3])
        savez_compressed('frame_nano.npz',frame_nano)

        bins=np.array([bins1,bins2,bins3])
        savez_compressed('bins.npz',bins)
        
        #processing and collating the data
        #collating the 3 beam data into one mega array
        INT_un=np.array([I1,I2,I3])  #unclean unnormalised mega array
        #declaring the normalisation
        INT_combined1=1/((I1**2+I2**2+I3**2)**(1/2))
        #removing any large values from the normalisation
        INT_combined1[INT_combined1== inf] = 1 #normalisaion array
        #normalising the mega array with the normalisation
        INT_n0=INT_un*INT_combined1
        INT_n=INT_n0.T  #unclean normalised mega array
        #saving the 3 arrays from above process in .npz format on disk
        savez_compressed('INT_combined1.npz',INT_combined1)
        savez_compressed('INT_un.npz',INT_un)
        savez_compressed('INT_n.npz',INT_n)
        
        
    def Single_Channel_Cluster(self,
               INT_n,
               i):
        """
        We perform HDBSCAN clustering on single channels of frequency.
        
        Parameters
        ----------
        INT_n : str
             Path to the normalised array having data from all 3 beams. 
        i : int
             Channel number on which HDBSCAN analysis will be perfomed on.
           
        Returns
        ----------     
        Plot : jpg
             Clustered plot obtained from HDBSCAN on single channels.
            
        """
        
        self.INT_n=INT_n
        self.i=i
        
        #Load the numpy .npz arrays
        INT_n=np.load('INT_n.npz')
        INT_n=INT_n['arr_0']
        
        #Perform HDBSCAN on single channels of frequencies
        start=time.time()
        clusterer=hdbscan.HDBSCAN(min_cluster_size=17,
                                  min_samples=16,
                                  cluster_selection_epsilon=0.01,
                                  allow_single_cluster=False,
                                  cluster_selection_method='eom')
        clusterer.fit(INT_n[:,self.i,:])
        INT_n_part=INT_n[:,self.i,:]
        INT_2d=clusterer.labels_
        INT_prob=clusterer.probabilities_
        INT_exemp_part=clusterer.exemplars_
        n_clusters_=len(set(INT_2d)) -(1 if -1 in INT_2d else 0)
        end=time.time()
        print('Time taken in sec:',end-start)
        print('No of clusters detected by HDBSCAN',n_clusters_)
        
        print('Plotting the scattering plots in the 3 dimensional intensoty space')
        
        get_ipython().run_line_magic('matplotlib', 'inline')
        plt.figure(figsize=(5,5))


        color_palette = sns.color_palette('Paired',n_clusters_)
        cluster_colors = [color_palette[x] if x >= 0
                          else (0.5, 0.5, 0.5)
                          for x in INT_2d]
        cluster_member_colors = [sns.desaturate(x, p) for x, p in
                                 zip(cluster_colors, 
                                     clusterer.probabilities_)]
        
        plt.xlabel('INT0')
        plt.ylabel('INT1')
        plt.title('HDBSCAN on single channels.')
        plt.scatter(INT_n_part[:,0],
                    INT_n_part[:,1],
                    marker=".",
                    alpha=1,
                    c=cluster_member_colors) #color='b')
        figname='HDBSCAN_on_single_channel_{}.jpg'.format(i)
        plt.savefig(figname)
        
        
    def Full_Channel_Cluster(self,
            INT_n,
            INT_combined1,
            method):
        """
        The following algorithm is used for 'ADD' method:
        1) Do clustering and save the exemplars.
        2) ADD the nearest exemplar from the normalized 3 beam data.
        3) Multiply the normalization with subtracted data.
        
        The following algorithm is used for 'SUB' method:
        1) Do clustering and save the exemplars.
        2) Subtract the nearest exemplar from the normalized 3 beam data.
        3) Multiply the normalization with subtracted data.
        
        More details is in Thesis.
        
        Parameters
        ----------
        INT_n : str
             Path to the normalised array having data from all 3 beams. 
        INT_combined1 : str
             Path to the normalising array for all 3 beams. 
        method : str, str value: 'ADD' or 'SUB'
             Method to be used while performing the data analysis.
             
        Returns
        ----------
        INT_prob1 : npz file
             Probabiliies array obtained from HDBSCAN for all 3 beams.
        INT_2d : npz file
             Labels array for all 3 beams which HDBSCAN generates.
        INT_exemp : npz file
             Exemplars which are obtained for all 3 beams from HDBSCAN clustering.
        INT_new : npz file
             Ceaned unnormalised array obtained at the end of the processing.
        INTnewnorm_wole : npz file
             Cleaned and normalised array obtained finally.
                 
        """
        
        self.INT_n=INT_n
        self.INT_combined1=INT_combined1
        self.method=method

        INT_n=np.load('INT_n.npz')
        INT_n=INT_n['arr_0']

        INT_combined1=np.load('INT_combined1.npz')
        INT_combined1=INT_combined1['arr_0']


        #entire 16384 channels HDBSCAN
        INT_prob1=[]  #probability
        INT_2d=[]     #labels
        INT_exemp=[]  #exemplars
        for i in range(0,INT_n.shape[1]):
            clusterer=hdbscan.HDBSCAN(min_cluster_size=20,
                                      min_samples=None)
            clusterer.fit(INT_n[:,i,:])
            INT_pred=clusterer.labels_
            n_clusters_=len(set(INT_pred)) -(1 if -1 in INT_pred else 0)
            INT_pred=clusterer.labels_
            INT_prob=clusterer.probabilities_

            INT_exemp.append(clusterer.exemplars_)
            INT_2d.append(INT_pred)
            INT_prob1.append(INT_prob)


        INT_prob1=np.array(INT_prob1)
        INT_2d=np.array(INT_2d)
        INT_exemp=np.array(INT_exemp)

        savez_compressed('INT_prob1.npz',INT_prob1)
        savez_compressed('INT_2d.npz',INT_2d)
        savez_compressed('INT_exemp.npz',INT_exemp)

        
        #exemplar processing
        INT_n_part=INT_n
        INT_2d_part=INT_2d
        INT_exemp_part=INT_exemp
        INTnew=np.zeros_like(INT_n_part)     #the cleaned norm array
        ARGMIN=[]

        if self.method=='ADD':
            
            for i in range(0,INT_2d.shape[0]):
                for j in range(0,INT_2d.shape[1]):
                    INT_exemp_combined=np.vstack(INT_exemp_part[i])

                    label=INT_2d_part[i,j]
                    value=INT_n_part[j,i,:]

                    if label!=-1:

                        argmin=np.argmin(np.sum((value-INT_exemp_part[i][label])**2,axis=1))
                        INTnew[j,i,:]=value+(INT_exemp_part[i][label][argmin])

                    else:
                        argmin=np.argmin(np.sum((value-INT_exemp_combined)**2,axis=1))
                        INTnew[j,i,:]=value+INT_exemp_combined[argmin]

                    ARGMIN.append(argmin)    

            print(INTnew.shape)    
            savez_compressed('INTnew.npz',INTnew)


            #denormalise the array
            INTnewnorm_whole=(INTnew.T)/INT_combined1 #cleaned unnorm array
            savez_compressed('INTnewnorm_whole.npz',(INTnewnorm_whole))
            
        elif self.method=='SUB':
            for i in range(0,INT_2d.shape[0]):
                for j in range(0,INT_2d.shape[1]):
                    INT_exemp_combined=np.vstack(INT_exemp_part[i])

                    label=INT_2d_part[i,j]
                    value=INT_n_part[j,i,:]

                    if label!=-1:

                        argmin=np.argmin(np.sum((value-INT_exemp_part[i][label])**2,axis=1))
                        INTnew[j,i,:]=value-(INT_exemp_part[i][label][argmin])

                    else:
                        argmin=np.argmin(np.sum((value-INT_exemp_combined)**2,axis=1))
                        INTnew[j,i,:]=value-INT_exemp_combined[argmin]

                    ARGMIN.append(argmin)    


            print('The subtracted normalised array has shape:',INTnew.shape)
           
            #Save the normalised subtracted array
            print('Saving the subtracted normalised array')
            savez_compressed('INTnew.npz',INTnew)


            #denormalise the subtracted array
            print('Denormalising the normalised subtracted array')
            print('Saving the denormalised array')
            INTnewnorm_whole=(INTnew.T)/INT_combined1 #cleaned unnorm array
            savez_compressed('INTnewnorm_whole.npz',(INTnewnorm_whole))


    def Iautils(self, 
                INTnewnorm_whole, 
                INT_un, 
                Weight, 
                fpga0,
                fpgan,
                dm,
                beam):
        
        '''
         
        After the exemplar subtraction has been done, the subtracted array is denormalized
        using the normalization array we had saved in the beginning. Next on the denormalized array
        from all 3 beams we apply various routines from IAUTILS.
        The regular workflow on IAUTILS is as follows:
        1) Declare various constants and input parameters to be passed to cascade object in
        step 2.
        2) Make a cascade object containing data/ spectrum with cascade.py .
        3) Dedisperse the data in a cascade object.
        4) Subband the data in a cascade object.
        5) Apply various analysis routines to detect pulse, determine SNR, optimise DM,
        generate TIMESERIES on the spectra object. After dedispersion and subbanding a
        spectra object is created within the cascade object from spectra.py script that works in
        tandem with cascade.py
        '''
        
        
         
        #IAUTILS
        self.INT_un= INT_un
        self.INTnewnorm_whole= INTnewnorm_whole
        self.Weight=Weight
        self.fpga0=fpga0
        self.fpgan=fpgan
        self.dm=dm
        self.beam=beam
        
        INT_un=np.load('INT_un.npz')
        INT_un=INT_un['arr_0']
        
        INTnewnorm_whole=np.load('INTnewnorm_whole.npz')
        INTnewnorm_whole=INTnewnorm_whole['arr_0']
        
        Weight=np.load('Weight.npz')
        Weight=Weight['arr_0']
        
        fpga0=np.load('fpga0.npz')
        fpga0=fpga0['arr_0']

        fpgan=np.load('fpgan.npz')
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
            #TIMESERIES[TIMESERIES<0]=0
        get_ipython().run_line_magic('matplotlib', 'inline')
        plt.plot(TIMESERIES)  #[11000:13000])
        print(np.argmax(TIMESERIES))


        FinalData=event.beams[0].intensity
        savez_compressed('FinalData.npz',FinalData)
        savez_compressed('TIMESERIES.npz',TIMESERIES)


    def Waterfaller(self,
                    FinalData,
                    TIMESERIES,
                    start, 
                    end) :
        '''
        Make a waterfall plot of the dedispersed,  subbanded array of
        data from the spectra object which is created within the cascade object
        '''
        
        self.FinalData=FinalData
        self.start=start
        self.end=end
        self.TIMESERIES=TIMESERIES
        
        FinalData=np.load('FinalData.npz')
        FinalData=FinalData['arr_0']
        
        TIMESERIES=np.load('TIMESERIES.npz')
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
        fig.savefig('Waterfall_Timeseries.jpg', dpi=300)
        
        spectra.find_burst((TIMESERIES[self.start:self.end]), 
                                  width_factor=10, 
                                  min_width=1, 
                                  max_width=10, 
                                  plot=True)
        
   
