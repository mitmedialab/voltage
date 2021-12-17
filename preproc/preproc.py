import tifffile as tiff
from libpreproc import preprocess_cython

def run_preprocessing(in_file, out_file, correction_file):
    in_image = tiff.imread(in_file).astype('float32')
    corrected, signal = preprocess_cython(in_image)
        
    tiff.imwrite(out_file, signal, photometric='minisblack')
    tiff.imwrite(correction_file, corrected, photometric='minisblack')
