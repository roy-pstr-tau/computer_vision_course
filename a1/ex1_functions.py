import numpy as np
import scipy.io
import random as rnd
import matplotlib as mplot
import time
import cv2
import PIL


def compute_homography_naive(mp_src, mp_dst):
    H, status = cv2.findHomography(np.transpose(mp_src), np.transpose(mp_dst))
    return H


# for section 4: im_dst = cv2.warpPerspective(im_src, h, (im_dst.shape[1],im_dst.shape[0]))


def test_homography(H, mp_src, mp_dst, max_err):

    im_out = cv2.warpPerspective(src=mp_src, M=H, dsize=(mp_dst.shape[1]+mp_src.shape[1], mp_dst.shape[0]+mp_src.shape[0]))
    mp_diff = np.sqrt((mp_dst[0] - im_out[0]) ** 2 + (mp_dst[1] - im_out[1]) ** 2)
    nof_inliers = len([i for i in mp_diff if i < max_err])
    fit_percent = nof_inliers / len(mp_diff)  # The probability (between 0 and 1) validly mapped src points (inliers)
    dist_mse = np.mean([i ** 2 for i in mp_diff if
                        i < max_err])  # Mean square error of the distances between validly mapped src points, to their corresponding dst points (only for inliers).
    return fit_percent, dist_mse


def compute_homography(mp_src, mp_dst, inliers_percent, max_err):
    H, status = cv2.findHomography(np.transpose(mp_src),
                                   np.transpose(mp_dst),
                                   method=cv2.RANSAC,
                                   ransacReprojThreshold=max_err,
                                   confidence=inliers_percent)
    return H


def panorama(img_src, img_dst, mp_src, mp_dst, inliers_percent, max_err):
    print('panorama')
