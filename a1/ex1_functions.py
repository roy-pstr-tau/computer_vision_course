import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import random as rnd
import matplotlib as mplot
import time
import cv2
import PIL.Image as Image


def compute_homography_naive(mp_src, mp_dst):
    print(np.transpose(mp_src))
    print(np.transpose(mp_dst))
    H, status = cv2.findHomography(np.transpose(mp_src), np.transpose(mp_dst))
    return H


def test_homography(H, mp_src, mp_dst, max_err):
    im_out = np.ndarray((mp_src.shape[0], mp_src.shape[1])).reshape(-1, 1, 2)
    print(im_out)
    print(np.transpose(mp_src).reshape(-1, 1, 2))
    cv2.perspectiveTransform(np.transpose(mp_src).reshape(-1, 1, 2), H, im_out)
    mp_diff = np.sqrt((np.transpose(mp_dst[0, :]) - im_out[:, 0, 0]) ** 2 + (np.transpose(mp_dst[1, :]) - im_out[:, 0, 1]) ** 2)
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
    H_ransac = compute_homography(mp_src, mp_dst, inliers_percent, max_err)
    show_panorama_image(H_ransac, img_src, img_dst)
    print('panorama')


def show_panorama_image(H, img_src, img_dst):
    p11, p12, p13, p14 = np.matmul(H, [0, 0, 1]), np.matmul(H, [img_src.shape[1] - 1, 0, 1]), np.matmul(
        H, [0, img_src.shape[0] - 1, 1]), np.matmul(H, [img_src.shape[1] - 1, img_src.shape[0] - 1, 1])
    p21, p22, p23, p24 = [0, 0], [img_dst.shape[1] - 1, 0], [0, img_dst.shape[0] - 1], [img_dst.shape[1] - 1,
                                                                                        img_dst.shape[0] - 1]
    dx_minus = (min(p11[0] / p11[2], p12[0] / p12[2], p13[0] / p13[2], p14[0] / p14[2], 0))
    dx_plus = (max(p11[0] / p11[2], p12[0] / p12[2], p13[0] / p13[2], p14[0] / p14[2], img_dst.shape[1] - 1))
    dy_minus = (min(p11[1] / p11[2], p12[1] / p12[2], p13[1] / p13[2], p14[1] / p14[2], 0))
    dy_plus = (max(p11[1] / p11[2], p12[1] / p12[2], p13[1] / p13[2], p14[1] / p14[2], img_dst.shape[0] - 1))
    newim_x_range = int(dx_plus - dx_minus)
    newim_y_range = int(dy_plus - dy_minus)
    H_offset = np.zeros((3, 3), dtype=float)

    if dx_minus < 0:
        H_offset[0][2] = -dx_minus
    if dy_minus < 0:
        H_offset[1][2] = -dy_minus
    H_offset[0][0] = 1
    H_offset[1][1] = 1
    H_offset[2][2] = 1

    im_out_src = cv2.warpPerspective(src=img_src, M=(np.matmul(H_offset, H)), dsize=(newim_x_range, newim_y_range))
    im_out_dst = cv2.warpPerspective(src=img_dst, M=H_offset, dsize=(newim_x_range, newim_y_range))
    im_out = im_out_dst + im_out_src
    plt.figure()
    plt.imshow(im_out)

    plt.axis('off')
    plt.show()
    return
