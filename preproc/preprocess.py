import os
import tifffile as tiff

def preprocess(inputname, mag):
    dir_path = os.path.dirname(os.path.realpath(__file__))
    if(mag == 16):
        ms = 3
        mp = 10
    elif(mag == 20):
        ms = 3
        mp = 15
    else:
        ms = 4
        mp = 20
    
    os.system('bash -c "cd ' + dir_path + ";./main -ms " + str(ms)+ " -mp " + str(mp) + " " + inputname + ' /tmp/pre_out/"')
    reg = tiff.imread('/tmp/pre_out/corrected.tif')
    signal = tiff.imread('/tmp/pre_out/signal.tif')
    
    return reg, signal
    
