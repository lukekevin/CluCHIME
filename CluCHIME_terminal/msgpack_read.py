"""
This code is a modified terminal based version of the original code from CHIME/FRB
used to read and  save the msgpack data chunks into numpy arrays
"""

import argparse
import time
import msgpack
import numpy as np
import zmq
from pathlib import Path
import glob 
import matplotlib.pyplot as plt
from numpy import savez_compressed 


class msgpack_reader:
    def __init__(self, msgpacked_chunk):
        c = msgpacked_chunk
        version = c[1]
        assert version in [1, 2]
        if version == 1:
            assert len(c) == 17
        if version == 2:
            assert len(c) == 21
        self.version = version
        compressed = c[2]
        compressed_size = c[3]
        self.beam = c[4]
        self.nupfreq = c[5]
        self.nt_per_packet = c[6]
        self.fpga_counts_per_sample = c[7]
        self.nt_coarse = c[8]
        self.nscales = c[9]
        self.ndata = c[10]
        self.fpga0 = c[11]
        self.fpgaN = c[12]
        self.binning = c[13]
        self.nt = self.nt_coarse * self.nt_per_packet

        scales = c[14]
        offsets = c[15]
        data = c[16]

        self.frame0_nano = 0
        self.nrfifreq = 0
        self.has_rfi_mask = False
        self.rfi_mask = None
        if self.version == 2:
            self.frame0_nano = c[17]
            self.nrfifreq = c[18]
            self.has_rfi_mask = c[19]
            mask = c[20]
            # to numpy
            mask = np.fromstring(mask, dtype=np.uint8)
            mask = mask.reshape((self.nrfifreq, self.nt // 8))
            # Expand mask!
            self.rfi_mask = np.zeros((self.nrfifreq, self.nt), bool)
            for i in range(8):
                self.rfi_mask[:, i::8] = (mask & (1 << i)) > 0

        if compressed:
            import pybitshuffle

            data = pybitshuffle.decompress(data, self.ndata)

        # Convert to numpy arrays
        self.scales = np.fromstring(scales, dtype="<f4")
        self.offsets = np.fromstring(offsets, dtype="<f4")
        self.scales = self.scales.reshape((-1, self.nt_coarse))
        self.offsets = self.offsets.reshape((-1, self.nt_coarse))
        self.data = np.frombuffer(data, dtype=np.uint8)
        self.data = self.data.reshape((-1, self.nt))

    def __str__(self):
        if self.has_rfi_mask:
            h, w = self.rfi_mask.shape
            masked = np.sum(self.rfi_mask == 0)
            rfistr = "yes, %i freqs, %i%% masked" % (
                self.nrfifreq,
                int(100.0 * masked / (h * w)),
            )
        else:
            rfistr = "no"
        return "AssembledChunk: beam %i, nt %i, fpga0 %i, rfi %s" % (
            self.beam,
            self.nt,
            self.fpga0,
            rfistr,
        )

    def decode(self):
        nf = self.data.shape[1]

        intensities = (
            self.offsets.repeat(self.nupfreq, axis=0).repeat(self.nt_per_packet, axis=1)
            + self.data
            * self.scales.repeat(self.nupfreq, axis=0).repeat(
                self.nt_per_packet, axis=1
            )
        ).astype(np.float32)

        weights = ((self.data > 0) * (self.data < 255)) * np.float32(1.0)

        return intensities, weights

    def time_start(self):
        # Nanoseconds per FPGA count
        fpga_nano = 2560
        return 1e-9 * (
            self.frame0_nano + self.fpga_counts_per_sample * fpga_nano * self.fpga0
        )

    def time_end(self):
        # Nanoseconds per FPGA count
        fpga_nano = 2560
        return 1e-9 * (
            self.frame0_nano
            + self.fpga_counts_per_sample * fpga_nano * (self.fpga0 + self.fpgaN)
        )


def read_msgpack_file(fn):
    f = open(fn, "rb")
    m = msgpack.unpackb(f.read())
    return msgpack_reader(m)


def unpack_data(fn):
    """
    Unpacks and de-compresses Intensity and Weights 
    Parameters
    ----------
    fn : filename of msgpack
    """

    chunk = read_msgpack_file(fn)
    intensity, weights = chunk.decode()
    frame0_nano = None
    nrfifreq = None
    rfi_mask = np.ones_like(intensity)
    version = chunk.version
    if version == 2:
        frame0_nano = chunk.frame0_nano
        nrfifreq = chunk.nrfifreq
        rfi_mask = chunk.rfi_mask
    return (
        intensity,
        weights,
        chunk.fpga0,
        chunk.fpgaN,
        chunk.binning,
        frame0_nano,
        nrfifreq,
        rfi_mask,
    )


def unpack_datafiles(fns, downsample=True):
    """
    Unpacks a list of filenames from a beam and appends to the
    list of intensities and weights. 
    """

    intensities = []
    weights = []
    fpga0s = []
    fpgaNs = []
    rfi_masks = []
    bin_list = []
    frame0_nanos = []

    for fn in fns:
        print(fn)
        (
            intensity,
            weight,
            fpga0,
            fpgaN,
            binning,
            frame0_nano,
            nrfifreq,
            rfi_mask,
        ) = unpack_data(fn)
        intensities.append(intensity)
        weights.append(weight)
        fpga0s.append(fpga0)
        fpgaNs.append(fpgaN)
        bin_list.append(binning)
        frame0_nanos.append(frame0_nano)
        rfi_masks.append(rfi_mask)

        stacked_intensities=np.hstack(intensities)[::-1]
        stacked_weights=np.hstack(weights)[::-1]
        stacked_rfi_mask=np.hstack(rfi_masks)[::-1]

    if len(set(bin_list)) == 1:  # if bin_list has only one unique item
        output_bin = bin_list[0]
        print("unpacked: ", output_bin)
        return stacked_intensities, stacked_weights, stacked_rfi_mask

"""
fpga0s,
fpgaNs,
output_bin,
np.hstack(rfi_masks)[::-1],
frame0_nanos)
"""

def statistical_graphs(stacked_intensities):
    
    plt.figure(figsize=(10,5))
    
    plt.subplot(3,1,1)
    plt.xlabel('Channels')
    plt.ylabel('St dev')
    plt.plot(np.std(stacked_intensities, axis=1))
    
    plt.subplot(3,1,2)
    plt.xlabel('Channels')
    plt.ylabel('median')
    plt.plot(np.median(stacked_intensities, axis=1))
    
    plt.subplot(3,1,3)
    plt.xlabel('Channels')
    plt.ylabel('mean')
    plt.plot(np.mean(stacked_intensities, axis=1))
    
    plt.savefig('stats_graph.png')
    

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    
    parser.add_argument('in_path', type=str,
        help='Path of the msgpack dir eg: /path/to/astro_1678xxxxx')
    args = parser.parse_args()
    in_path=args.in_path

    #Glob the .msgpack chunks from the given dir
    glob_path=glob.glob(in_path+'*.msgpack')
    glob_path.sort()
    
    print('Running the conversion now\n')
    
    #Run the msgpack reading analysis
    stacked_intensities,stacked_weights,stacked_rfi_mask=unpack_datafiles(glob_path)
    savez_compressed('intensity.npz', stacked_intensities)
    savez_compressed('weights.npz', stacked_weights)
    savez_compressed('rfi_mask.npz', stacked_rfi_mask)
    
    print('Finishing the conversion\n')
    
    #Additionaly save the stats of mean, median, std for the intensity data
    statistical_graphs(stacked_intensities)