# Voltage Imaging Data Processing


## Directory Structure
* preproc   : preprocessing (motion/shading correction and spike extraction)
* segment   : segmentation of cells
* demix     : demixing of voltage traces of individual cells
* simulation: synthetic data generation
* pipeline  : pipeline script
* lib       : libraries

## How to Run

$ cd lib/tiff-4.1.0_mod  
$ ./configure  
$ make  
$ cd ../../preproc  
$ make  
$ ./main <input_tiff> <output_path>

The preprocessing results will be
* <output_path>/corrected.tif: motion/shading corrected video
* <output_path>/signal.tif: extracted signal (spikes)
* ./motion.dat: motion estimation result

Building tiff-4.1.0_mod allows much faster saving of multipage tiffs, but one could also use standard libtiff instead.
