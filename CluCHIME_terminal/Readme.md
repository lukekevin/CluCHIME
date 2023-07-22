## Basic Usage of CluCHIME_terminal codes.
The set of codes here can be executed from terminal without the necessity of a notebook.
### Workflow of the codes
The following is the flow of the codes so that the data analysis is done properly. <br /> 

### 1) collate_cluster.py
The data pakcets in msgpack format from CHIME/FRB L1 nodes are send to this code which collats the data packets, extracts the numpy arrays from them, converts the extracted numpy arrays into npz format.Then the collated arrays are worked upon by the HDBSCAN clustering and further research related algorithms are applied and finally at the end of the execution the cleaned normalised and denormalised arrays are dumped at a directory. 
<br />
##### Usage
 `python collate_cluster.py --args`
 
### 2) cascade_subbanding.py 
The cleaned and uncleaned denormalised arrays are sent to this code and then the IAUTILS processes are applied on the array which makes a cascade object, dedisperses the arrays, subbands them and downsampled them and finally the subbanded, dedispersed, downsampled data array is dunmped along with it's timeseries as npz files.
<br />
##### Usage
`python cascade_subbanding.py --args`

### 3) cascade_waterfaller.py
The dedispersed, subbanded, dowmsapled data is loaded along with the timeseries and then the waterfall plots and the timeseries are saved as a jpg file.
<br />
##### Usage
`python cascade_waterfaller.py --args` 

### 4) msgpack_read.py
Read and convert a list of msgpack files for a given event from CHIME/FRB 
<br />
##### Usage
`python msgpack_read.py --args` 

