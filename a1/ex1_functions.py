#   ex1_functions
#   Roy Pasternak and Shany Amir
#   ids: 204219273  312545965
#
import numpy as np
import matplotlib.pyplot as plt
import random as rnd
import cv2
np.set_printoptions(suppress=True)

#   geometricDistance
#   Calculate the geometric distance between estimated points and original points

def geometricDistance(correspondence, h):
    p1 = np.transpose(np.matrix([correspondence[0].item(0), correspondence[0].item(1), 1]))
    estimatep2 = np.dot(h, p1)
    estimatep2 = (1 / estimatep2.item(2)) * estimatep2

    p2 = np.transpose(np.matrix([correspondence[0].item(2), correspondence[0].item(3), 1]))
    error = p2 - estimatep2
    return np.linalg.norm(error)

#   compute_homography_naive
#   return a 3x3 homography matrix
#   no dealing with outliers

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


#   test_homography
#   return number of inliers precent and mse calc between the mapped points to the destination points

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

#   compute_homography
#   return a 3x3 homography matrix
#   dealing with outliers using RANSAC algorithm

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

#   panorama
#   return panoramic image of 2 input images and their matching points
#   using RANSAC to deal with outliers
#   using backwards mapping (can be change to forward)

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

    H_offset = np.identity(3, dtype=float)
    if dx_minus < 0:
        H_offset[0][2] = -dx_minus
    if dy_minus < 0:
        H_offset[1][2] = -dy_minus
    M=np.matmul(H_offset, H).astype(np.float32)

    if mapping=="forward":
        img_out_src = forward_mapping(img_src, H=M, out_width=panorama_width, out_height=panorama_height)
    elif mapping=="backward":
        img_out_src = backward_mapping(img_src, H=M, out_width=panorama_width, out_height=panorama_height)
    else:
        assert(False)

    img_out_dst = backward_mapping(img_dst, H=H_offset, out_width=panorama_width, out_height=panorama_height)
    img_out = np.where(img_out_dst.round() == 0, img_out_src, img_out_dst)
    return img_out

#   backward_mapping
#   maps the src image according to 3x3 homography matrix
#   backward mapping using bilinear interpolation

def backward_mapping(img_src, H, out_width, out_height):
    return cv2.warpPerspective(src=img_src, M=H, dsize=(out_width, out_height), flags=cv2.INTER_LINEAR)

#   forward_mapping
#   forward mapping the src image according to 3x3 homography matrix

def forward_mapping(img_src, H, out_width, out_height):
    points_in_src_in_panorma = repeat_product(np.arange(img_src.shape[1]), np.arange(img_src.shape[0]))
    points_forwarded_to_mapped_source = forward(points_in_src_in_panorma, H)
    points_forwarded_to_mapped_source = points_forwarded_to_mapped_source.reshape(img_src.shape[0], img_src.shape[1], 3)

    # points_forwarded_to_mapped_source : mapping from [x,y] in panorama -> [y,x,1] in source
    im_out_src = np.zeros((out_height, out_width, 3), dtype=np.uint8)
    for p in points_in_src_in_panorma:
        x_in_pan = p[1]
        y_in_pan = p[0]
        points_in_mapped_src = points_forwarded_to_mapped_source[x_in_pan, y_in_pan]
        if 0 <= points_in_mapped_src[0] < im_out_src.shape[1] and 0 <= points_in_mapped_src[1] < im_out_src.shape[0]:
            im_out_src[points_in_mapped_src[1], points_in_mapped_src[0]] = img_src[x_in_pan, y_in_pan]
    return im_out_src

#   forward
#   calculates H*src_point

def forward(src_points, h):
    mapped_src_points = np.matmul(h, src_points.T).T
    number_of_points = mapped_src_points.shape[0]
    mapped_src_points = mapped_src_points / mapped_src_points[:, 2].reshape(number_of_points,1)
    return mapped_src_points.round().astype(int)

#   show_panorama_image
#   for our use, used to display src image after forward mapping with 3x3 Homography martix

def show_panorama_image(H, img_src, img_dst):
    # 2d corners of src:
    corners_src = np.array([[0, 0], [img_src.shape[1] - 1, 0], [0, img_src.shape[0] - 1], [img_src.shape[1] - 1, img_src.shape[0] - 1]])
    # adding the 'z' dim from (x,y) -> (x,y,1) for each corner point:
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
    #im_out_src = cv2.warpPerspective(src=img_src, M=(np.matmul(H_offset, np.float32(H))),
    #                                 dsize=(panorama_width, panorama_height), flags=cv2.INTER_LINEAR)
    im_out_src = forward_mapping(img_src, (np.matmul(H_offset, H).astype(np.float32)), panorama_width, panorama_height)
    plt.figure()
    plt.imshow(im_out_src)

    plt.axis('off')
    plt.show()
    return

#   repeat_product
#   rearrange (x, y) points in a column

def repeat_product(x, y):
    points_2d = np.transpose([np.tile(x, len(y)),np.repeat(y, len(x))])
    return np.hstack((points_2d, np.ones((points_2d.shape[0], 1), dtype=np.int)))


