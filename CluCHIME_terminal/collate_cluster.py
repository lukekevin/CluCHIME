'''
Author: Kevin Luke 
CluCHIME.py was made for MSc thesis project at TIFR, Mumbai
Date created: 11 April 2022
Date last modified: 12 April  2022
'''

import argparse
import numpy as np
from numpy import savez_compressed 
from iautils.conversion import chime_intensity
from numpy import inf
import glob 
import hdbscan
import matplotlib.pyplot as plt
import seaborn as sns


''' 
THIS IS VERSION OF CODE CAN BE EXECUTED DIRECTLY FROM A TERMINAL
For running this code successfully you will
need NUMPY, SCIPY, Scikit-Learn, IAUTILS and frb-common utils preinstalled.
The later 2 libraries are closed CHIME/FRB libraries from GITHUB. The code 
is slow and not efficiet in terms of memory usage and needs to updated 
accordingly in future. The average run times for CLUSTERING ANALYSIS on 
full data is ~6 hrs on  Dr. Shriharsh Tendulkar's workstation PRITHVI 
at TIFR which has 128 GB ram, 14 TB of disk space. On regular PC's
the run times will be much higher.
'''
    
def collate(path1, 
            path2,
            path3
            out_dir):
    ''' 
    Prepping the DATA.

    The individual .msg datapackets has shape 16384,1024 
    which corresponds to 16384 frequency channels between 400 to 800 MHZ and 1024 
    timestamps which corresponds to roughly 1 second of observation.For each 
    beam there will be series of such .msgpack data packets leading to 
    shape of 16384, n*1024 where n corresponds to no of individual data pakcets
    and seconds of observation. If there are 10 seconds of data then the n=10  
    packets are there etc. The mega array whether normalised or unnormalised has final 
    shape of 16384,n*1024,3 where 3 corresponds to 3 beams for which this CluCHIME 
    library was developed.
    '''
    
    filelist1=glob.glob(path1)
    filelist1.sort()
    (I1,Weight1,
     bins1,fpga01st,
     fpgan1st,frame_nano1,
     rfi_mask1)=chime_intensity.unpack_datafiles(filelist1)


    filelist2=glob.glob(path2)
    filelist2.sort()
    (I2,Weight2,
     bins2,fpga02nd,
     fpgan2nd,frame_nano2,
     rfi_mask2)=chime_intensity.unpack_datafiles(filelist2)


    filelist3=glob.glob(path3)
    filelist3.sort()
    (I3,Weight3,
     bins3,fpga03rd,
     fpgan3rd,frame_nano3,
     rfi_mask3)=chime_intensity.unpack_datafiles(filelist3)

    #Savingthe metadata from .msgpack data    
    #metadata
    Weight=np.array([Weight1,Weight2,Weight3])
    savez_compressed(out_dir+'Weight.npz',Weight)

    fpgan=np.array([fpgan1st,fpgan2nd,fpgan3rd])
    savez_compressed(out_dir+'fpgan.npz',fpgan)

    fpga0=np.array([fpga01st,fpga02nd,fpga03rd])
    savez_compressed(out_dir'+'fpga0.npz',fpga0)

    rfi_mask=np.array([rfi_mask1,rfi_mask2,rfi_mask3,])
    savez_compressed(out_dir+'rfi_mask.npz',rfi_mask)

    frame_nano=np.array([frame_nano1,frame_nano2,frame_nano3])
    savez_compressed(out_dir+'frame_nano.npz',frame_nano)

    bins=np.array([bins1,bins2,bins3])
    savez_compressed(out_dirh+'bins.npz',bins)

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
    savez_compressed(out_dir+'INT_combined1.npz',INT_combined1)
    savez_compressed(out_dir+'INT_un.npz',INT_un)
    savez_compressed(out_dir+'INT_n.npz',INT_n)
    
    print('Shape of unnormalised array')
    print(INT_un.shape)
    print('Shape of normalised array')
    print(INT_n.shape)
    
    '''
    returning the array so they can be used directly when they are required 
    in other parts of code
    '''
    return INT_un, INT_n,INT_combined1
    
def Single(INT_n,i):
    '''
    We perform HDBSCAN clustering on single channels of frequency.
    We feed the unnormalised intensity array at INT_un and
    the normalised intensity array INT_n and the normalisation array INT_combined.
    Here we perform HDBSCAN clustering on the normalised data from Prepdata.
    Labels and Probaility are also  stored from  and they are stored.
    '''

   #Perform HDBSCAN on single channels of frequencies
    clusterer=hdbscan.HDBSCAN(min_cluster_size=17,
                              min_samples=16,
                              cluster_selection_epsilon=0.01,
                              allow_single_cluster=True)
    
    clusterer.fit(INT_n[:,i,:])
    INT_2d=clusterer.labels_
    INT_prob=clusterer.probabilities_
    INT_exem=clusterer.exemplars_
    n_clusters_=INT_2d.max() + 1
    print('No of clusters detected by HDBSCAN',n_clusters_)

    plt.figure(figsize=(5,5))

    color_palette = sns.color_palette('Paired',n_clusters_)
    cluster_colors = [color_palette[x] if x >= 0
                      else (0.5, 0.5, 0.5)
                      for x in INT_2d]
    cluster_member_colors = [sns.desaturate(x, p) for x, p in
                             zip(cluster_colors, clusterer.probabilities_)]
    plt.xlabel('INT0')
    plt.ylabel('INT1')
    plt.title('HDBSCAN on single frequencies')
    plt.scatter(INT_n[:,i,0],
                INT_n[:,i,1],
                marker=".",
                alpha=1,
                c=cluster_member_colors)
    plt.savefig(out_dir+'clustered_plot.jpg')


    
    
def Sub(INT_n,
        INT_combined1,
        out_dir):
    '''
    Here we use Cluster exemplar subtraction method.
    We feed the unnormalised intensity array at INT_un and
    the normalised intensity array INT_n and the normalisation array INT_combined.
    Here we perform HDBSCAN clustering on the normalised data from Prepdata. 
    For HDBSCAN I dont provide the option to change the min cluster size and min sample size 
    parameters as those defined parameters give the better clustering for this data.
    Labels and Probaility are also  stored from  and they are stored. Exemplars 
    are also stored from HDBSCAN. Then the following algorithm is used:
    1) Do clustering and save the exemplars.
    2) Subtract the nearest exemplar from the normalized 3 beam data.
    3) Multiply the normalization with subtracted data.
    More details is in Thesis.
    '''
    
   #Run HDBSCAN on entire 16384 channels
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

    #convert the python lists from HDBSCAN to numpy arrays
    INT_prob1=np.array(INT_prob1)
    INT_2d=np.array(INT_2d)
    INT_exemp=np.array(INT_exemp) 


    #save the arrays as comprressed numpy files, .npz files
    savez_compressed(out_dir+'INT_prob1.npz',INT_prob1)
    savez_compressed(out_dir+'INT_2d.npz',INT_2d)
    savez_compressed(out_dir+'INT_exemp.npz',INT_exemp)

    #Exemplar subtraction algorithm
    #Just renaming the arrays to be fed in the for loop 
    INT_n_part=INT_n
    INT_2d_part=INT_2d
    INT_exemp_part=INT_exemp
    INTnew=np.zeros_like(INT_n_part)     #the cleaned norm array
    ARGMIN=[]
    #The ACTUAL SUBTRACTION ALGORITHM
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

    print('Time taken for code run in sec:',end-start)
    print('The subtracted normalised array has shape:',INTnew.shape)

    #Save the normalised subtracted array
    print('Saving the subtracted normalised array')
    savez_compressed(out_dir+'INTnew.npz',INTnew)


    #denormalise the subtracted array
    print('Denormalising the normalised subtracted array')
    print('Saving the denormalised array')
    INTnewnorm_whole=(INTnew.T)/INT_combined1 #cleaned unnorm array
    savez_compressed(out_dir+'INTnewnorm_whole.npz',(INTnewnorm_whole))

    
    
def Add(INT_n,
        INT_combined1,
        outdir):
    '''
    Here we use Cluster exemplar addition method.
    We feed the unnormalised intensity array at INT_un and
    the normalised intensity array INT_n and the normalisation array INT_combined.
    Here we perform HDBSCAN clustering on the normalised data from Prepdata.
    Labels and Probaility are also  stored from  and they are stored. Exemplars 
    are also stored from HDBSCAN. Then the following algorithm is used:
    1) Do clustering and save the exemplars.
    2) ADD the nearest exemplar from the normalized 3 beam data.
    3) Multiply the normalization with subtracted data.
    More details is in Thesis.
    '''
    
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

        savez_compressed(out_dir+'INT_prob1.npz',INT_prob1)
        savez_compressed(out_dir+'INT_2d.npz',INT_2d)
        savez_compressed(out_dir+'INT_exemp.npz',INT_exemp)

        #exemplar subtraction
        INT_n_part=INT_n
        INT_2d_part=INT_2d
        INT_exemp_part=INT_exemp
        INTnew=np.zeros_like(INT_n_part)     #the cleaned norm array
        ARGMIN=[]

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
        savez_compressed(out_dir+'INTnew.npz',INTnew)

        #denormalise the array
        INTnewnorm_whole=(INTnew.T+1)/INT_combined1 #cleaned unnorm array
        savez_compressed(out_dir+'INTnewnorm_whole.npz',(INTnewnorm_whole))


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--pathname1',
                        type=str, 
                        help='Data path for glob process where .msgpack for beam 1 is stored')
    parser.add_argument('--pathname2',
                        type=str, 
                        help='Data path for glob process where .msgpack for beam 2 is stored')
    parser.add_argument('--pathname3', 
                        type=str, 
                        help='Data path for glob process where .msgpack for beam 3 is stored')
    parser.add_argument('--out_dir_path', 
                        type=str, 
                        help='Path where the processed data will be dumped')
    parser.add_argument('--range', 
                        type=str, 
                        help='choose clustering for either single/full 16k channels')
    parser.add_argument('--technique', 
                        type=str,
                        help=' if chosen for full channels then choose add/sub method')
    
    args = parser.parse_args()
    
    pathname1=args.pathname1
    pathname2=args.pathname2
    pathname3=args.pathname3
    out_dir_path=args.out_path_dir
    
    out_dir = out_dir_path + 'cluchime_clustering_code' + "/"
    os.makedirs(out_dir, exist_ok=True)
    
    INT_un,INT_n,INT_combined1=collate(pathname1,
                                      pathname2,
                                      pathname3,
                                      out_dir)
    if args.range=='single':
        print('Input frequency channel number from 0-16384')
        freq_channel=int(input())
        Single(INT_n,
               freq_channel,
               out_dir)
        
    elif args.range=='full':
         print('Running clustering and subsequent process over full data')
        if args.technique=='sub':
           print('Choosing Sub method for full data processing')
           print('See theses, contact me for more information on these two methods')
           Sub(INT_n,
                INT_combined1,
                out_dir)
            
        elif args.technique=='add':
             print('Choosing Add method for full data processing')
             print('See theses, contact me for more information on these two methods')
             Add(INT_n,
                INT_combined1,
                out_dir)
            
            
