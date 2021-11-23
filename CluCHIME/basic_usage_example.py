from iautils.conversion import chime_intensity as ci
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np

from sklearn.neighbors import LocalOutlierFactor
from sklearn.datasets import make_blobs
from numpy import quantile, where, random
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from sklearn.cluster import DBSCAN
import time
from numpy import asarray
from numpy import savetxt 
from sklearn.cluster import OPTICS, cluster_optics_dbscan
import hdbscan
import seaborn as sns
from numpy import save
from numpy import savez_compressed 

import datetime, time
import numpy as np
import matplotlib.pyplot as plt
from frb_common import common_utils
from iautils import cascade, spectra, spectra_utils
from iautils.conversion import chime_intensity

from numpy import inf



import glob 




function='CLUSTERUTILS'
REM=CluCHIME(function)
Pathname1='/DATA/shriharsh/CHIME/multibeam_ML/astro_155769173/intensity/raw/1153/astro*.msgpack'
Pathname2='/DATA/shriharsh/CHIME/multibeam_ML/astro_155769173/intensity/raw/2153/astro*.msgpack'
Pathname3='/DATA/shriharsh/CHIME/multibeam_ML/astro_155769173/intensity/raw/3153/astro*.msgpack'
REM.Prepdata(Pathname1,Pathname2,Pathname3)
INT_un='INT_un.npz'
INT_n='INT_n.npz
INT_combined1='INT_combined1.npz
REM.Sub(INT_un,INT_n,INT_combined1)
INT_un='INT_un.npz'
INT_n='INT_n.npz'
i=2694
REM.Single(INT_un,INT_n,i)
INT_un='INT_un.npz'
INT_n='INT_n.npz
INT_combined1='INT_combined1.npz
REM.Add(INT_un,INT_n,INT_combined1)
INT_un='INT_un.npz'
INTnewnorm_whole='INT_newnorm_whole.npz'
Weight='Weight.npz'
fpga0='fpga0.npz'
fpgan='fpgan.npz'
dm=558.5
beam='3153CLEAN'
REM.Iautils(INT_un,INTnewnorm_whole,Weight,fpga0,fpgan,dm,beam)
FinalData='FinalData.npz'
TIMESERIES='TIMESERIES.npz'
start=12200
end=12400
REM.Waterfaller(FinalData,TIMESERIES,start,end)
