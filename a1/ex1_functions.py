import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import random as rnd
import matplotlib as mplot
import time
import cv2
import PIL.Image as Image


#
#  Calculate the geometric distance between estimated points and original points

def geometricDistance(correspondence, h):
    p1 = np.transpose(np.matrix([correspondence[0].item(0), correspondence[0].item(1), 1]))
    estimatep2 = np.dot(h, p1)
    estimatep2 = (1 / estimatep2.item(2)) * estimatep2

    p2 = np.transpose(np.matrix([correspondence[0].item(2), correspondence[0].item(3), 1]))
    error = p2 - estimatep2
    return np.linalg.norm(error)


def compute_homography_naive(mp_src, mp_dst):
    eq_system = []
    for point in range(mp_src.shape[1]):
        x = mp_src[0, point]
        y = mp_src[1, point]
        x_tag = mp_dst[0, point]
        y_tag = mp_dst[1, point]
        a1 = [x, y, 1, 0, 0, 0, -(x_tag * x), -(x_tag * y), -x_tag]
        a2 = [0, 0, 0, x, y, 1, -(y_tag * x), -(y_tag * y), -y_tag]
        eq_system.append(a1)
        eq_system.append(a2)
    matrixEq = np.matrix(eq_system)

    #  Singular Value Decomposition
    u, s, v = np.linalg.svd(matrixEq)

    # reshape the min singular value into a 3 by 3 matrix
    H = np.reshape(v[8], (3, 3))

    # normalize and now we have h
    H = (1 / H.item(8)) * H
    return H


def test_homography(H, mp_src, mp_dst, max_err):
    im_out = np.ndarray((mp_src.shape[0], mp_src.shape[1])).reshape(-1, 1, 2)
    cv2.perspectiveTransform(np.transpose(mp_src).reshape(-1, 1, 2), H, im_out)
    mp_diff = np.sqrt(
        (np.transpose(mp_dst[0, :]) - im_out[:, 0, 0]) ** 2 + (np.transpose(mp_dst[1, :]) - im_out[:, 0, 1]) ** 2)
    nof_inliers = len([i for i in mp_diff if i < max_err])
    fit_percent = nof_inliers / len(mp_diff)  # The probability (between 0 and 1) validly mapped src points (inliers)
    dist_mse = np.mean([i ** 2 for i in mp_diff if
                        i < max_err])  # Mean square error of the distances between validly mapped src points, to their corresponding dst points (only for inliers).
    return fit_percent, dist_mse


def compute_homography(mp_src, mp_dst, inliers_percent, max_err):
    maxInliers = []
    finalH = None
    for i in range(1000):
        #  find 4 random points to calculate a homography
        a, b, c, d = rnd.randrange(0, mp_src.shape[1]), rnd.randrange(0,
                                                                                  mp_src.shape[1]), rnd.randrange(
            0, mp_src.shape[1]), rnd.randrange(0, mp_src.shape[1])
        rnd_src = [[mp_src[0, a], mp_src[0, b], mp_src[0, c], mp_src[0, d]],
                   [mp_src[1, a], mp_src[1, b], mp_src[1, c], mp_src[1, d]]]
        rnd_dst = [[mp_dst[0, a], mp_dst[0, b], mp_dst[0, c], mp_dst[0, d]],
                   [mp_dst[1, a], mp_dst[1, b], mp_dst[1, c], mp_dst[1, d]]]

        #  call the homography function on those points
        h = compute_homography_naive(np.asarray(rnd_src), np.asarray(rnd_dst))
        inliers = []

        for i in range(mp_src.shape[1]):
            d = geometricDistance(np.matrix([mp_src[0, i], mp_src[1, i], mp_dst[0, i], mp_dst[1, i]]), h)
            if d < max_err:
                inliers.append([mp_src[0, i], mp_src[1, i], mp_dst[0, i], mp_dst[1, i]])

        if len(inliers) > len(maxInliers):
            maxInliers = inliers
            finalH = h

        if (len(maxInliers) / (mp_src.shape[1])) > inliers_percent:
            break
    return finalH


def panorama(img_src, img_dst, mp_src, mp_dst, inliers_percent, max_err):
    print('panorama')


def show_panorama_image(H, img_src, img_dst):
    p11, p12, p13, p14 = np.matmul(np.asarray(H), [0, 0, 1]), np.matmul(np.asarray(H),
                                                                        [img_src.shape[1] - 1, 0, 1]), np.matmul(
        np.asarray(H), [0, img_src.shape[0] - 1, 1]), np.matmul(np.asarray(H),
                                                                [img_src.shape[1] - 1, img_src.shape[0] - 1, 1])
    p21, p22, p23, p24 = [0, 0], [img_dst.shape[1] - 1, 0], [0, img_dst.shape[0] - 1], [img_dst.shape[1] - 1,
                                                                                        img_dst.shape[0] - 1]
    p11 = p11 / p11.item(2)
    p12 = p12 / p11.item(2)
    p13 = p13 / p11.item(2)
    p14 = p14 / p11.item(2)

    dx_minus = (min(p11.item(0), p12.item(0), p13.item(0), p14.item(0), 0))
    dx_plus = (max(p11.item(0), p12.item(0), p13.item(0), p14.item(0), img_dst.shape[1] - 1))
    dy_minus = (min(p11.item(1), p12.item(1), p13.item(1), p14.item(1), 0))
    dy_plus = (max(p11.item(1), p12.item(1), p13.item(1), p14.item(1), img_dst.shape[0] - 1))
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
    im_out_src = cv2.warpPerspective(src=img_src, M=(np.matmul(H_offset, np.float32(H))),
                                     dsize=(newim_x_range, newim_y_range))
    im_out_dst = cv2.warpPerspective(src=img_dst, M=H_offset, dsize=(newim_x_range, newim_y_range))
    im_out = im_out_dst + im_out_src
    plt.figure()
    plt.imshow(im_out)

    plt.axis('off')
    plt.show()
    return
