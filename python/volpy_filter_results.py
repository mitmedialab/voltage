import os
import numpy as np
import tifffile
from scipy import ndimage

# Once volpy_pipeline.py has been ran, this script filters out the ROIs in which neurons have been detected


BASE_DIR="/home/yves/Projects/active/Fixstars/datasets/VolPy_dataset/"
ALL_RESULTS_DIR=BASE_DIR+"/"+"voltage_rois/"
if not os.path.exists(ALL_RESULTS_DIR):
    os.mkdir(ALL_RESULTS_DIR)

spikes_threshold=15 # From how many spikes do we consider a neuron as active?

for NAME in sorted(os.listdir(BASE_DIR)):
    directory = BASE_DIR+"/"+NAME
    gt_rois_file = directory+"/"+NAME+"_ROI.zip"
    results_file = f"{directory}/volpy_{NAME}_adaptive_threshold.npy"
    image_file = f"{directory}/{NAME}.tif"
    result_dir = f"{directory}/voltage_rois/"
    if not os.path.exists(image_file):
        continue
    if not os.path.exists(result_dir):
        os.mkdir(result_dir)
    print(NAME)
    results = np.load(results_file, allow_pickle=True).item()
    failed=tot=0
    polygons=list()
    multipage=list()
    for loc, roi, nums, ls in zip(results["locality"], results["ROIs"], results["num_spikes"], results["low_spikes"]):
        if ls or not loc:
            failed+=1
        else:
            # multi-page tiff format
            multipage.append(roi)

            # R-CNN npz format
            # img2 = ndimage.binary_dilation(roi, [[False, True, False], [True, True, True], [False, True, False]])
            # contour = img2.astype(float) - roi
            # (ys, xs) = contour.nonzero()
            # polygon = {'name': 'polygon', 'all_points_x': xs, 'all_points_y': ys}
            # polygons.append(polygon)
            # np.savez(f"{result_dir}/rois.npz", polygons)
            # np.savez(f"{ALL_RESULTS_DIR}/{NAME}_rois.npz", polygons)
        tot+=1
    if len(multipage)>0:
        data = np.stack(multipage).astype('uint8')
        tifffile.imwrite(f"{ALL_RESULTS_DIR}/{NAME}_rois.tif", data, photometric='minisblack')
    else:
        open(f"{ALL_RESULTS_DIR}/{NAME}_rois.tif", "w").close()
        print(f"Warning! {ALL_RESULTS_DIR}/{NAME}_rois.tif is empty")
    print(f"{failed} on {tot} failed")