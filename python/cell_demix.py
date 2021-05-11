from numba import jit
import time
import gc
import numpy as np
import cv2
import multiprocessing
from skimage.morphology import convex_hull_image
from copy import deepcopy
from postprocess import postprocess
import nvgpu
from joblib import Parallel, delayed

### Cleaning NMF data
min_area = 30
max_area = 900

kernel = np.ones((5,5),np.uint8)

class roi:
    def __init__(self, ID, Nid):
        self.ID = ID
        self.Nid = Nid
    
    def assign(self, mask, idx):
        self.mask = np.array(mask)
        self.idx = np.array(idx)
    
    def assign_sig(self, sig):
        self.sig = np.array(sig)
            

def parallel_process_NMfout(ctr, Wblis, vid):
    rois = []
    nctr = 0
    Wb = Wblis[ctr]

    params = cv2.SimpleBlobDetector_Params()
    TIME_FRAMES = len(vid)
    
    # Change thresholds
    params.minThreshold = 0
    params.maxThreshold = 255


    # Filter by Circularity
    params.filterByCircularity = True
    params.minCircularity = 0.5

    #     # Filter by Convexity
    params.filterByConvexity = True
    params.minConvexity = 0.6

    # Filter by Inertia
    params.filterByInertia = True
    params.minInertiaRatio = 0.01
    
    # Create a detector with the parameters
    detector = cv2.SimpleBlobDetector_create(params)
    
    for k in range(len(Wb)):
        img = np.array(Wb[k])
        img = img - img.min()
        img = img / img.max()
        img[img < 0.6] = 0
        img = img * 255
        img = img.astype('uint8')
        thresed = img
        ret, markers = cv2.connectedComponents(thresed)
        total_contours = markers.max()
        lis_area = []
        for i in range(0, total_contours+1):
            check = np.array(thresed[markers == i])
            val = int(check.sum()/255)
            lis_area.append(val)
        rem_idx = [index for index,value in enumerate(lis_area) if ((value < min_area) or (value > max_area))]

        for i in rem_idx:
            markers[markers == i] = 0
        total_contours = len(np.unique(markers))

        for i in np.unique(markers):
            if (i == 0):
                continue
            image = markers.astype('uint8')
            image[markers!=i]=0
            image[markers==i]=255

            im = cv2.bitwise_not(image)
            keypoints = detector.detect(im)
            if(len(keypoints) > 0):
                r = roi(nctr, ctr)
                nctr = nctr + 1               

                image = image / 255
                image = image.astype('uint8')

                fmask = image.astype('float32')
                idx = np.array(np.where(image==1))
                idx = idx.transpose((1,0))

                r.assign(image, idx)
                rois.append(r)

    return rois

def find_NMF_size_req(ns, nf, nc):
    return int(np.ceil((2*ns*nf + 5*ns*nc + 6*nf*nc + 2*nc*nc + ns + nf + 6)/1000000.0))

def get_NMF_runs(X, N):
    s_req = find_NMF_size_req(X.shape[0], X.shape[1], N[-1])
    
    g_info = nvgpu.gpu_info()
    ng = len(g_info)
    s = []
    for i in range(ng):
        s.append(g_info[i]['mem_total'] - g_info[i]['mem_used'])

    r = 0
    run = {}
    tN = N
    while True:

        if(len(tN) == 0):
            break
        nN = []
        run[r] = []
        rm = np.zeros(ng)
        for i in range(ng):
            rm[i] = s[i]
        for i in range(len(tN)):
            n = tN[i]
            j = i % ng
            if(s_req < rm[j]):
                run[r].append((n, j))
                rm[j] -= s_req
            else:
                nN.append(n)
        tN = nN
        r += 1
    return run

def compute_NMF(X, n, ypred, gpu_n):
    import os
    from nmf_wrapper import NMF
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_n)
    model = NMF(n_components=n, init='random', random_state=0, max_iter=500, method=1)
    W = model.fit_transform(X)
    Wb = W.transpose((1,0))
    Wb = np.reshape(Wb, (n, ypred.shape[1], ypred.shape[2]))
    return Wb, n

def postprocess_NMF(file, frames):
    return postprocess(file, frames)

def demix_neurons(file, ypred, en):
    file = file - file.min()
    file = file / file.max()
    vid = file
    TIME_FRAMES = len(file)
    X = np.reshape(ypred, (ypred.shape[0], ypred.shape[1]*ypred.shape[2]))
    X = X.transpose((1,0))
    
    Nbar = en
    if (Nbar < 5):
        N = list(range(max(1, Nbar-1), Nbar + 1))
    else:
        N = list(range(Nbar-2, Nbar+2))
    # Nbar = en
    # if (Nbar < 5):
    #     N = list(range(1, Nbar + 2))
    # else:
    #     N = list(range(Nbar-4, Nbar+6))
    tic = time.time()
    runs = get_NMF_runs(X, N)
    Wblis = []
    Wbd = {}
    for r in runs:
        Wblis = Parallel(n_jobs=len(runs[r]))(delayed(compute_NMF)(X, n, ypred, gpu_n) for n, gpu_n in runs[r])
        for ret in Wblis:
            Wbd[ret[1]] = ret[0]
    Wblis = []
    for n in N:
        Wblis.append(Wbd[n])
    print("NMF len:", len(Wblis), runs)
    # Wblis = Parallel(n_jobs=len(N))(delayed(compute_NMF)(X, n, ypred, int(n%2)) for n in N)
    
    time_NMF = time.time() - tic
    print("Finished computing NMF")


    def parallel_masking(x):
        return cv2.multiply(fmask, vid[x]).sum()

    tic = time.time()

    NMFout = []
    for ctr in range(len(N)):
        NMFout.append(parallel_process_NMfout(ctr, Wblis, file))
    frames = []
    for rs in NMFout:
        for r in rs:
            frames.append(r.mask)
    frames = np.array(frames, dtype='float32')
    if(len(frames) > 0):
        sigs = postprocess_NMF(file, frames)
    else:
        print("No frames to process. No neuron detected after NMF!")
        sigs = []
    
    idx = 0
    for rs in NMFout:
        for r in rs:
            r.sig = sigs[idx]
            idx += 1

    time_NMFProcess = time.time() - tic
    
    bkup = deepcopy(NMFout)
    
    @jit(nopython=True)
    def compute_distances(X1, X2):
        dist = np.sqrt(np.sum(np.square(X1), axis=1) + np.sum(np.square(X2), axis=1) - 2 * np.sum(np.multiply(X1, X2), axis=1))
        return dist

    def min_dist_ROI(R1, R2):
        l1 = len(R1)
        l2 = len(R2)
        A = np.zeros((l1*l2, 2))
        B = np.zeros((l1*l2, 2))

        a = np.array(R1)
        b = np.array(R2)
        for i in range(l2):
            A[i*l1:(i+1)*l1] = a
            a = np.roll(a, 1, axis=0)
        for i in range(l1):
            B[i*l2:(i+1)*l2] = b
        min_d = compute_distances(A, B).min()
        del A, B, a, b
        gc.collect()
        return min_d

    def overlap_area(R1, R2):
        i1 = np.where(R1 > 0)
        i1 = [(i1[0][i], i1[1][i]) for i in range(len(i1[0]))]
        i2 = np.where(R2 > 0)
        i2 = [(i2[0][i], i2[1][i]) for i in range(len(i2[0]))]

        i3 = [i for i in i1 if i in i2]

        a1 = R1.sum()/R1.max()
        a2 = R2.sum()/R2.max()
        a3 = len(i3)

        return int(round(max(min((a3/a1),1), min((a3/a2), 1))*100))

    def overlap_min_area(R1, R2):
        i1 = np.where(R1 > 0)
        i1 = [(i1[0][i], i1[1][i]) for i in range(len(i1[0]))]
        i2 = np.where(R2 > 0)
        i2 = [(i2[0][i], i2[1][i]) for i in range(len(i2[0]))]

        i3 = [i for i in i1 if i in i2]

        a1 = R1.sum()/R1.max()
        a2 = R2.sum()/R2.max()
        a3 = len(i3)

        return int(round(min(min((a3/a1),1), min((a3/a2), 1))*100))



    NMFout = deepcopy(bkup)

    def check_roi_in_list(ls, r):
        for kroi in ls:
            if(overlap_area(kroi.mask, r.mask) > 50):
                return True
        return False

    rem_list = []
    keep_list = []
    for k in range(len(NMFout)):
        multinmf = []
        for k_roi_ctr in range(len(NMFout[k])):
            k_roi = NMFout[k][k_roi_ctr]
            for i in range(k+1, len(NMFout)):
                coverlaps = []
                overlaps = []
                for j_roi in NMFout[i]:
                    if(overlap_area(k_roi.mask, j_roi.mask) > 80 and overlap_min_area(k_roi.mask, j_roi.mask) > 60):
                        coverlaps.append(j_roi)
                    elif(overlap_area(k_roi.mask, j_roi.mask) > 60):
                        C = np.corrcoef(k_roi.sig, j_roi.sig)[0][1]
                        if(C > 0.85):
                            coverlaps.append(j_roi)
                flag = 0
                for j1 in range(len(coverlaps)-1):
                    o1_roi = coverlaps[j1]
                    flag = 0
                    for j2 in range(j1+1, len(coverlaps)):
                        o2_roi = coverlaps[j2]
                        C = np.corrcoef(o1_roi.sig, o2_roi.sig)[0][1]
                        if(C < 0.85):
                            flag = 1
                            break
                    if (flag == 1):
                        break
                if(flag == 1):
                    rem_list.append(k_roi)
                else:
                    for o_roi in coverlaps:
                        rem_list.append(o_roi)

    for r_roi in rem_list:
        Nctr = r_roi.Nid
        idx = -1
        for i in range(len(NMFout[Nctr])):
            if(NMFout[Nctr][i].ID == r_roi.ID):
                idx = i
                break
        if(idx != -1):
            del NMFout[Nctr][idx]

    total_roi = []
    for k in range(len(NMFout)):
        for j in range(len(NMFout[k])):
            if(check_roi_in_list(total_roi, NMFout[k][j]) == False):
                total_roi.append(NMFout[k][j])
            else:
                pass

    img = np.array(np.average(vid, axis=0))
    img = img - img.min()
    img = img / img.max()
    img = (img).astype('float32')
    for troi in total_roi:
        edge = cv2.Canny(troi.mask, 0, 1)
        idx = np.where(edge == 255)
        y = idx[0][0]
        x = idx[1][0]

        edge = (cv2.bitwise_not(edge)/255).astype('float32')
        img = cv2.multiply(img, edge)

    # Store files
    neu_data = {}

    for i in range(0, len(total_roi)):
        idxx = [total_roi[i].idx[j][0] - 1 for j in range(len(total_roi[i].idx))]
        idxy = [total_roi[i].idx[j][1] - 1 for j in range(len(total_roi[i].idx))]
        neu_data[str("neuron_" + str(i))] = {"ID":str(total_roi[i].ID), "NID":str(total_roi[i].Nid), "idxx":str(list(idxx)), "idxy":str(list(idxy))}
    
    demix_data = {}
    demix_data['mask'] = img
    demix_data['info'] = neu_data
    demix_data['NMF_time'] = time_NMF
    demix_data['NMFProcess_time'] = time_NMFProcess
    print("NMF:", time_NMF, "NMFProcess:", time_NMFProcess)

    return demix_data
    
