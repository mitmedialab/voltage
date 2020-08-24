import tifffile as tiff
import preprocess
import time
import sys
mp = {}
mp['level']=2
mp['search_size']=3
mp['patch_size']=10
mp['patch_offset']=7
mp['x_range']=0.7
mp['y_range']=1.0
mp['a_stdev']=1.0
mp['m_stdev']=3.0
mp['thresh_xy']=1.0
mp['length']=2000
mp['thresh_c']=0.4
mp['is_motion_correction'] = 1
# A = tiff.imread('/u/ramdas/Voltage_Imaging/Training_data/OldData/holder/WholeTifs/018.tif')
A = tiff.imread(sys.argv[1])
A = A.astype('float32')
#tiff.imsave('input.tif', A)
tic = time.time()
B = preprocess.preprocess(A, mp)
toc = time.time()
print(B.shape)
#tiff.imsave('test.tif', B)
print("End of program:", round((toc-tic), 2))
