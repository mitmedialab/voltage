# 2022 Fixstars, Yves Quemener
#
# A few functions designed to help produce meaningful outputs when manipulating voltage imaging results


from scipy import ndimage
import numpy as np
import cv2


# Draws the perimeter of ROIs in an image
#
# img: nd-array image. Either grayscale or RGB
# areas: list if boolean nd-array of the same size as img, describing ROIs
# color: color the contour should be
def add_contours(img, areas, color=[0.3, 0.2, 1]):
    # Make RGB
    if len(img.shape) == 2 or img.shape[2] == 1:
        out_img = np.stack([img, img, img], axis=2)
    else:
        out_img = img

    for area in areas:
        # Compute contour
        img2 = ndimage.binary_dilation(area, [[False, True, False], [True, True, True], [False, True, False]])
        contour = img2.astype(float) - area
        # Add the contour
        out_img[contour == True] = color
    return out_img


# Creates a list that maps indices from `areas1` into indices from `areas2`
# `areas1` and `areas2` are lists of 2D boolean np-arrays
def match_areas(areas1, areas2):
    results_lookup = np.zeros((100,100))
    mapping = list()

    for i, area in enumerate(areas2):
        results_lookup[area==True]=i+1


    for i, area in enumerate(areas1):

        intersection = np.logical_and(area>0, results_lookup>0)
        inter_area = np.count_nonzero(intersection)
        inter_id = int(np.median(results_lookup[intersection]))
        if inter_area<10:
            inter_id=-1
        mapping.append(inter_id)
    return mapping


# Draws numbers in an image on the center of areas
# img: image to be drawn on. nd-array describing either a grayscale or RGB image
# areas: list of boolean nd-array describing areas. Only used to find the position to draw the text at
# nums: list of numbers to be displayed on the corresponding area center
def show_numbers(img, areas, nums, zoom_factor=4):
    font = cv2.FONT_HERSHEY_SIMPLEX
    color = (255, 0, 0)
    thickness = 1
    out_size = img.shape[0] * zoom_factor, img.shape[1] * zoom_factor

    # Make RGB
    if len(img.shape) == 2 or img.shape[2] == 1:
        out_img = np.stack([img, img, img], axis=2)
    else:
        out_img = img
    # Make bigger without looking the pixelated appearance
    out_img = cv2.resize(out_img, out_size, interpolation=cv2.INTER_NEAREST)

    for area, num in zip(areas, nums):
        pos = (int(np.median(np.nonzero(area)[1])) * zoom_factor - 10,
               int(np.median(np.nonzero(area)[0])) * zoom_factor)
        out_img = cv2.putText(out_img, str(num), pos, font,
                              0.35, color, thickness, cv2.LINE_AA)
    return out_img