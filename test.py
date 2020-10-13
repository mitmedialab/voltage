import tifffile as tiff
import preprocess
import time
import sys
import numpy as np
import nvgpu

def get_minium_size_for_gpu():
    gpu_info = nvgpu.gpu_info()
    sizes = []
    SAFE_MEM = 1500
    for gi in gpu_info:
        sizes.append((gi['mem_total'] - gi['mem_used']))
    return min(sizes) - SAFE_MEM

def chunkIt(seqlen, num):
    avg = seqlen / float(num)
    out = []
    last = 0.0
    si = []
    ei = []
    while last < seqlen:
        si.append(int(last))
        ei.append(int(last + avg))
        last += avg
    return si, ei

def size_required(file):
    T = file.shape[0]
    H = max(128, file.shape[1])
    W = max(128, file.shape[2])
    return int((T * H * W * 4 * 2)/1e6)

param = {}
idname = '027'
param[idname] = {}
param[idname]['filename'] = '/home/ramdas/Documents/Voltage_Imaging/WholeTifs/' + idname + '.tif'
param[idname]['output_dimension'] = 128
param[idname]['magnification'] = 16
param[idname]['expected_neurons'] = 5
param[idname]['is_motion_correction'] = 1
param[idname]['motion_correction'] = {}
param[idname]['motion_correction']['search_size']=3
param[idname]['motion_correction']['patch_size']=10
param[idname]['motion_correction']['patch_offset']=7



A = tiff.imread(param[idname]['filename'])
A = A.astype('float32')
T = len(A)
gsize = get_minium_size_for_gpu()
print(sys.getsizeof(A))
print(sys.getsizeof(A[0:44000]))
print(sys.getsizeof(np.array(A[0:44000], 'float32')))
tic = time.time()
size_req = size_required(A)
if(size_req > gsize):
    si, ei = chunkIt(T, (size_req//gsize) + 1)
    mc = []
    pd = []
    # print(si, ei)
    for i in range(len(si)):
        s = si[len(si) - i - 1]
        e = ei[len(si) - i - 1]
        # print(s, e)
        param[idname]['motion_correction']['length'] = (e-s)
        print(s, e)
        m, p = preprocess.preprocess(np.array(A[s:e], 'float32'), param[idname])
        time.sleep(5)
        mc.append(m)
        pd.append(p)
    B = np.concatenate(mc, axis=0)
    C = np.concatenate(pd, axis=0)
else:
    param[idname]['motion_correction']['length'] = len(A)
    B, C = preprocess.preprocess(A, param[idname])


toc = time.time()
print(C.shape)
# tiff.imsave('test.tif', B[:100])


# D = []
# for i in range(0, len(A), 1000):
#   param['018']['motion_correction']['length'] = 1000
#   B, C = preprocess.preprocess(A[i:i+1000], param['018'])
#   D.append(B)
# D = np.concatenate(D, axis=0)
# toc = time.time()
# print(D.shape)
# tiff.imsave('test.tif', D[:100])


tiff.imsave('proc.tif', C[:100])
print("End of program:", round((toc-tic), 2))