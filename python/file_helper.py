import tifffile as tiff
import numpy as np
import h5py
import pathlib

def read_hdf5(fname):
    with h5py.File(fname, 'r') as f:
        for key in f.keys():
            item = f[key]
            if(isinstance(item, h5py.Dataset) == True):
                return item[()]

def write_hdf5(fname, data):
    with h5py.File(fname, 'w') as f:
        dset = f.create_dataset("default", data=data)

def fread(fname):

    ext = pathlib.Path(fname).suffix

    if(ext == '.tif'):
        data = tiff.imread(fname)
    elif(ext == '.npy'):
        data = np.load(fname)
    elif(ext == '.hdf5'):
        data = read_hdf5(fname)
    else:
        raise Exception("Unsupported File format!")
    return data

def fwrite(fname, data):

    ext = pathlib.Path(fname).suffix

    if(ext == '.tif'):
        data = tiff.imsave(fname, data)
    elif(ext == '.npy'):
        data = np.save(fname, data)
    elif(ext == '.hdf5'):
        data = write_hdf5(fname, data)
    else:
        raise Exception("Unsupported File format!")