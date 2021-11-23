# CluCHIME
This is a Python library for handling radio astronomical      data from 3 beams/detectors from CHIME/FRB instrument and      performing CLUSTERING ANALYSIS and RFI removal on the multi      beam data .


The data format used is .msgpack which is the format      used for the INTENSITY data generated by the CHIME/FRB pipeline.      
However with minimal modifications to inputs the library can also      be used for multi beam data from any radio interferometric telescope.     


The CluCHIME library and the tools here were made during my MSc thesis      at TATA INSTITUTE OF FUNDAMENTAL RESEARCH under guidance of Dr. Shriharsh      Tendulkar, Reader, DEPARTMENT OF ASTRONOMY AND ASTROPHYSICS. 
Nikhil Londhe      of IIT GANDHINAGAR helped me understand array manipulation in NUMPY.     

The library is in primitive phase now and in future will be updated      accordinlgy by me. The results obtained from the CLUSTERING ANALYSIS      are mixed. 

This CluCHIME library also streamlines and eases the various     relevant data  analysis from IAUTILS Python library. 


The IAUTILS library was developed by Dr. Shriharsh Tendulkar et al      at CHIME/FRB Collaboration. For running CluCHIME library successfully you will     need NUMPY, SCIPY, Scikit-Learn, IAUTILS and frb-common utils preinstalled.     

The later 2 libraries are closed CHIME/FRB libraries from GITHUB. 

The code      is slow and not efficiet in terms of memory usage and needs to updated      accordingly in future. The average run times for CLUSTERING ANALYSIS on      full data is ~6 hrs on  Dr. Shriharsh Tendulkar's workstation PRITHVI      at TIFR which has 128 GB ram, 14 TB of disk space. On regular PC's   the run times will be much higher.
