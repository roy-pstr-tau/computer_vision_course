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
    assert(mp_src.shape[1]==mp_dst.shape[1])
    number_of_points=mp_src.shape[1]
    for point in range(number_of_points):
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
    return np.asarray(H)


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
        a, b, c, d = rnd.randrange(0, mp_src.shape[1]), rnd.randrange(0, mp_src.shape[1]), rnd.randrange(
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


def panorama(img_src, img_dst, mp_src, mp_dst, inliers_percent, max_err, mapping="backward"):
    H = compute_homography(mp_src, mp_dst, inliers_percent, max_err)

    # 2d corners of dst and src:
    corners_dst = np.array([[0, 0], [img_dst.shape[1] - 1, 0], [0, img_dst.shape[0] - 1], [img_dst.shape[1] - 1, img_dst.shape[0] - 1]])
    corners_src = np.array([[0, 0], [img_src.shape[1] - 1, 0], [0, img_src.shape[0] - 1], [img_src.shape[1] - 1, img_src.shape[0] - 1]])
    # adding the 'z' dim from (x,y) -> (x,y,1) for each corner point:
    corners_3d_dst = np.hstack((corners_dst, np.ones((4, 1), dtype=np.int)))
    corners_3d_src = np.hstack((corners_src, np.ones((4, 1), dtype=np.int)))
    # casting the source corners to the dst coord system using H
    corners_src_to_panorama = np.matmul(H, corners_3d_src.T).T
    # normalize the (x,y) values of each corner using the z value.
    corners_src_to_panorama_normalized = corners_src_to_panorama / corners_src_to_panorama[:, 2].reshape(4, 1)

    # deltas of x and y
    dx_minus = np.min(np.hstack((corners_src_to_panorama_normalized[:, 0], 0)))
    dx_plus = np.max(np.hstack((corners_src_to_panorama_normalized[:, 0], img_dst.shape[1] - 1)))
    dy_minus = np.min(np.hstack((corners_src_to_panorama_normalized[:, 1], 0)))
    dy_plus = np.max(np.hstack((corners_src_to_panorama_normalized[:, 1], img_dst.shape[0] - 1)))

    # calc the final panorama image dimensions (after the stitching!)
    panorama_width = int(dx_plus - dx_minus)
    panorama_height = int(dy_plus - dy_minus)

    # calc the corners of the source image after cast to panaroma coords:
    corners_of_src_in_panorama = corners_src_to_panorama_normalized + np.array([-dx_minus, -dy_minus, 0])

    if mapping=="forward":
        img_out_src = forward_mapping(img_src, H=H, out_width=panorama_width, out_height=panorama_height, x_offset=dx_minus, y_offset=dy_minus)
    elif mapping=="backward":

        print("sainty check:")
        print("original:", corners_3d_src)
        print("backward calculated:", np.apply_along_axis(backward, 1, corners_of_src_in_panorama, H, -dx_minus, -dy_minus))
        img_out_src = backward_mapping(img_src, H=H, out_width=panorama_width, out_height=panorama_height, x_offset=dx_minus, y_offset=dy_minus, corners=corners_of_src_in_panorama)
    else:
        assert(False)

    #img_out_dst = cv2.warpPerspective(src=img_dst, M=H_offset, dsize=(panorama_width, panorama_height), flags=cv2.INTER_LINEAR)
    # TODO should it backward mapped?? i dont think so... we apply only offset
    img_out_dst = forward_mapping(img_dst, H=np.identity(3, dtype=np.uint8), out_width=panorama_width, out_height=panorama_height, x_offset=dx_minus, y_offset=dy_minus)
    img_out = np.where(img_out_dst == 0, img_out_src, img_out_dst)
    # mean_img = cv2.addWeighted(im_out_dst, 0.5, im_out_src, 0.5, 0)
    # zeros_img = np.zeros(img_out_dst.shape, dtype=np.int)
    # img_out = np.where(((img_out_src != 0) & (img_out_dst != 0)), mean_img, zeros_img) + \
    #           np.where(img_out_src == 0, img_out_dst, zeros_img) + \
    #           np.where(img_out_src == 0, img_out_dst, zeros_img)

    plt.figure()
    plt.imshow(img_out)
    plt.title(str(mapping+" mapping"))
    plt.axis('off')
    plt.show()

def forward_mapping(img_src, H, out_width, out_height, x_offset, y_offset):
    # calc the offset matrix, two possible offsets which are dx_minus and dy_minus
    H_offset = np.identity(3, dtype=float)
    if x_offset < 0:
        H_offset[0][2] = -x_offset
    if y_offset < 0:
        H_offset[1][2] = -y_offset

    return cv2.warpPerspective(src=img_src, M=(np.matmul(H_offset, np.float32(H))),
                               dsize=(out_width, out_height), flags=cv2.INTER_LINEAR)

def backward_mapping(img_src, H, out_width, out_height, x_offset, y_offset, corners):
    # # 2d corners of dst and src:
    # corners_dst = np.array([[0, 0], [img_dst.shape[1] - 1, 0], [0, img_dst.shape[0] - 1], [img_dst.shape[1] - 1, img_dst.shape[0] - 1]])
    # corners_src = np.array([[0, 0], [img_src.shape[1] - 1, 0], [0, img_src.shape[0] - 1], [img_src.shape[1] - 1, img_src.shape[0] - 1]])
    # # adding the 'z' dim from (x,y) -> (x,y,1) for each corner point:
    # corners_3d_dst = np.hstack((corners_dst, np.ones((4, 1), dtype=np.int)))
    # corners_3d_src = np.hstack((corners_src, np.ones((4, 1), dtype=np.int)))
    # # casting the source corners to the dst coord system using H
    # corners_src_to_panorama = np.matmul(H, corners_3d_src.T).T
    # # normalize the (x,y) values of each corner using the z value.
    # corners_src_to_panorama_normalized = corners_src_to_panorama / corners_src_to_panorama[:, 2].reshape(4, 1)
    # # deltas of x and y
    # dx_minus = np.min(np.hstack((corners_src_to_panorama_normalized[:, 0], 0)))
    # dx_plus = np.max(np.hstack((corners_src_to_panorama_normalized[:, 0], img_dst.shape[1] - 1)))
    # dy_minus = np.min(np.hstack((corners_src_to_panorama_normalized[:, 1], 0)))
    # dy_plus = np.max(np.hstack((corners_src_to_panorama_normalized[:, 1], img_dst.shape[0] - 1)))
    #
    # # calc the final panorama image dimensions (after the stitching!)
    # panorama_width = int(dx_plus - dx_minus)
    # panorama_height = int(dy_plus - dy_minus)

    src_width_in_panorma = np.floor(np.max(corners[:, 0])).astype(np.int)
    src_height_in_panorma = np.floor(np.max(corners[:, 1])).astype(np.int)
    points_in_src_in_panorma = repeat_product(np.arange(src_width_in_panorma), np.arange(src_height_in_panorma))
    print("Calculating backward mapping...")
    points_backwarded_to_source = np.apply_along_axis(backward, 1, points_in_src_in_panorma, H, -x_offset, -y_offset) # TODO think how to make this faster!
    points_backwarded_to_source = points_backwarded_to_source.reshape(src_height_in_panorma,src_width_in_panorma,3)
    # points_backwarded_to_source : mapping from [x,y] in panorama -> [y,x,1] in source
    im_out_src = np.zeros((out_height, out_width, 3), dtype=np.uint8)
    print("Calculating backward interpolation...")
    for p in points_in_src_in_panorma:
        x_in_pan = p[1]
        y_in_pan = p[0]
        point_in_src = points_backwarded_to_source[x_in_pan, y_in_pan]
        im_out_src[x_in_pan, y_in_pan, :] = bilinear_inter(img_src, x=point_in_src[1], y=point_in_src[0])

    return im_out_src
    # TODO calc also in backward?
    # im_out_dst = cv2.warpPerspective(src=img_dst, M=H_offset, dsize=(panorama_width, panorama_height),
    #                                  flags=cv2.INTER_LINEAR)
    #
    # # TODO remove the mean! jsut take the dest pixels!
    # mean_img = cv2.addWeighted(im_out_dst, 0.5, im_out_src, 0.5, 0)
    # im_out = np.where(((im_out_src != 0) & (im_out_dst != 0)), mean_img,
    #                   np.zeros(im_out_dst.shape, dtype=np.int)) + np.where(im_out_src == 0, im_out_dst,
    #                                                                        np.zeros(im_out_dst.shape,
    #                                                                                 dtype=np.int)) + np.where(
    #     im_out_dst == 0, im_out_src, np.zeros(im_out_dst.shape, dtype=np.int))
    # plt.figure()
    # plt.imshow(im_out)

def backward(dst, h, dx, dy):
    H_inv = np.linalg.inv(h)
    offsets = np.array([dx, dy, 0])
    dst = dst - offsets
    src = np.matmul(H_inv, dst.T).T
    src = src / src[2]
    return src

def repeat_product(x, y):
    points_2d = np.transpose([np.tile(x, len(y)),np.repeat(y, len(x))])
    return np.hstack((points_2d, np.ones((points_2d.shape[0], 1), dtype=np.int)))

def bilinear_inter(src, x, y):
    '''

    :param src: image
    :param x: float pixel index
    :param y: float pixel index
    :return: bilinear interpolation of the pixels: [(u,v), (u+1,v),(u,v+1),(u+1,v+1)] where u = floor(x), v = floor(u)
    '''
    if x < 0 or y < 0:
        return np.zeros((3,), dtype=np.uint8)
    if np.ceil(x) >= src.shape[0] or np.ceil(y) >= src.shape[1]:
        return np.zeros((3,), dtype=np.uint8)
    u = np.floor(x).astype(np.int)
    v = np.floor(y).astype(np.int)
    alpha = x-u
    beta = y-v
    alpha_vec = np.array([1 - alpha, alpha])
    beta_vec = np.array([1 - beta, beta]).T
    interpolate_value = []
    for c in range(src.shape[2]): # TODO vectorize!
        curr_channel = src[:,:,c]
        points_matrix = np.array([
                                    [curr_channel[u, v], curr_channel[u, v+1]],
                                    [curr_channel[u+1, v], curr_channel[u+1, v+1]]
                                ])

        interpolate_value.append(np.matmul(alpha_vec,np.matmul(points_matrix, beta_vec)))

    return np.array(interpolate_value, dtype=np.uint8)

def show_panorama_image(H, img_src, img_dst):
    # 2d corners of dst and src:
    corners_dst = np.array([[0, 0], [img_dst.shape[1] - 1, 0], [0, img_dst.shape[0] - 1], [img_dst.shape[1] - 1, img_dst.shape[0] - 1]])
    corners_src = np.array([[0, 0], [img_src.shape[1] - 1, 0], [0, img_src.shape[0] - 1], [img_src.shape[1] - 1, img_src.shape[0] - 1]])
    # adding the 'z' dim from (x,y) -> (x,y,1) for each corner point:
    corners_3d_dst = np.hstack((corners_dst, np.ones((4, 1), dtype=np.int)))
    corners_3d_src = np.hstack((corners_src, np.ones((4, 1), dtype=np.int)))
    # casting the source corners to the dst coord system using H
    corners_src_to_panorama = np.matmul(H, corners_3d_src.T).T
    # normalize the (x,y) values of each corner using the z value.
    corners_src_to_panorama_normalized = corners_src_to_panorama/corners_src_to_panorama[:,2].reshape(4,1)

    # deltas of x and y
    dx_minus = np.min(np.hstack((corners_src_to_panorama_normalized[:, 0], 0)))
    dx_plus = np.max(np.hstack((corners_src_to_panorama_normalized[:, 0], img_dst.shape[1] - 1)))
    dy_minus = np.min(np.hstack((corners_src_to_panorama_normalized[:, 1], 0)))
    dy_plus = np.max(np.hstack((corners_src_to_panorama_normalized[:, 1], img_dst.shape[0] - 1)))

    # calc the final panorama image dimensions (after the stitching!)
    panorama_width = int(dx_plus - dx_minus)
    panorama_height = int(dy_plus - dy_minus)

    # calc the offset matrix, two possible offsets which are dx_minus and dy_minus
    H_offset = np.identity(3, dtype=float)
    if dx_minus < 0:
        H_offset[0][2] = -dx_minus
    if dy_minus < 0:
        H_offset[1][2] = -dy_minus

    # cast source and dst images to the panorama image plane:
    im_out_src = cv2.warpPerspective(src=img_src, M=(np.matmul(H_offset, np.float32(H))),
                                     dsize=(panorama_width, panorama_height), flags=cv2.INTER_LINEAR)
    im_out_dst = cv2.warpPerspective(src=img_dst, M=H_offset, dsize=(panorama_width, panorama_height), flags=cv2.INTER_LINEAR)

    # choosing the pixels from the source image only where the dst image pixels are zero! that our way of stitching.
    im_out = np.where(im_out_dst == 0, im_out_src, im_out_dst)
    # mean_img = cv2.addWeighted(im_out_dst, 0.5, im_out_src, 0.5, 0)
    # im_out = np.where(((im_out_src != 0) & (im_out_dst != 0)), mean_img,
    #                   np.zeros(im_out_dst.shape, dtype=np.int)) + np.where(im_out_src == 0, im_out_dst,
    #                                                                        np.zeros(im_out_dst.shape,
    #                                                                                 dtype=np.int)) + np.where(
    #     im_out_dst == 0, im_out_src, np.zeros(im_out_dst.shape, dtype=np.int))
    # # im_out = im_out_dst + im_out_src
    plt.figure()
    plt.imshow(im_out)

    plt.axis('off')
    plt.show()
    return