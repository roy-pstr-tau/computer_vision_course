import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import scipy.io
import time

from ex1_functions import *


def tic():
    return time.time()


def toc(t):
    return float(tic()) - float(t)


##########################################################
# Don't forget to fill in your IDs!!!
# students' IDs:
ID1 = 312545965
ID2 = 204219273
##########################################################


# Parameters
max_err = 25
inliers_percent = 0.8

# Read the data:
img_src = mpimg.imread('src.jpg')
img_dst = mpimg.imread('dst.jpg')
# matches = scipy.io.loadmat('matches') #matching points and some outliers
matches = scipy.io.loadmat('matches_perfect')  # loading perfect matches
match_p_dst = matches['match_p_dst'].astype(float)
match_p_src = matches['match_p_src'].astype(float)

# Compute naive homography
tt = time.time()
H_naive = compute_homography_naive(match_p_src, match_p_dst)
print('Naive Homography {:5.4f} sec'.format(toc(tt)))
print(H_naive)

p11, p12, p13, p14 = np.matmul(H_naive, [0, 0, 1]), np.matmul(H_naive, [img_src.shape[0]-1, 0, 1]), np.matmul(H_naive, [0, img_src.shape[1]-1, 1]), np.matmul(H_naive, [img_src.shape[0]-1, img_src.shape[1]-1, 1])
p21, p22, p23, p24 = [0, 0], [img_dst.shape[0]-1, 0], [0, img_dst.shape[1]-1], [img_dst.shape[0]-1, img_dst.shape[1]-1]
print("points:")
print(p11, p12, p13, p14, p21, p22, p23, p24)
print("points:")
dx_minus = (min(p11[0]/p11[2], p12[0]/p12[2], p13[0]/p13[2], p14[0]/p14[2], 0))
dx_plus = (max(p11[0]/p11[2], p12[0]/p12[2], p13[0]/p13[2], p14[0]/p14[2], img_dst.shape[0]-1))
dy_minus = (min(p11[1]/p11[1], p12[1]/p12[1], p13[1]/p13[1], p14[1]/p14[1], 0))
dy_plus = (max(p11[1]/p11[1], p12[1]/p12[1], p13[1]/p13[1], p14[1]/p14[1], img_dst.shape[1]-1))
H_offset = [[0, 0, dx_minus], [0, 0, dy_minus], [0, 0, 1]]
newim_x_range = int(dx_plus - dx_minus)
newim_y_range = int(dy_plus - dy_minus)
if dx_minus<0:
    H_naive[0][2] = H_naive[0][2] - dx_minus
if dy_minus<0:
    H_naive[1][2] = H_naive[1][2] - dy_minus

im_out = cv2.warpPerspective(src=img_src, M=H_naive, dsize=(newim_x_range, newim_y_range))
# im_out[0:img_dst.shape[0], 0:img_dst.shape[1]] = img_dst
plt.figure()
plt.imshow(im_out)

plt.axis('off')
plt.show()

# Test naive homography
tt = time.time()
fit_percent, dist_mse = test_homography(H_naive, match_p_src, match_p_dst, max_err)
print('Naive Homography Test {:5.4f} sec'.format(toc(tt)))
print([fit_percent, dist_mse])

# Compute RANSAC homography
tt = tic()
H_ransac = compute_homography(match_p_src, match_p_dst, inliers_percent, max_err)
print('RANSAC Homography {:5.4f} sec'.format(toc(tt)))
print(H_ransac)

# Test RANSAC homography
tt = tic()
fit_percent, dist_mse = test_homography(H_ransac, match_p_src, match_p_dst, max_err)
print('RANSAC Homography Test {:5.4f} sec'.format(toc(tt)))
print([fit_percent, dist_mse])

# Build panorama
tt = tic()
img_pan = panorama(img_src, img_dst, match_p_src, match_p_dst, inliers_percent, max_err)
print('Panorama {:5.4f} sec'.format(toc(tt)))

plt.figure()
panplot = plt.imshow(img_pan)
plt.title('Great Panorama')
# plt.show()


## Student Files
# first run "create_matching_points.py" with your own images to create a mat file with the matching coordinates.
max_err = 25  # <<<<< YOU MAY CHANGE THIS
inliers_percent = 0.8  # <<<<< YOU MAY CHANGE THIS

img_src_test = mpimg.imread('src_test.jpg')
img_dst_test = mpimg.imread('dst_test.jpg')

matches_test = scipy.io.loadmat('matches_test')

match_p_dst = matches_test['match_p_dst']
match_p_src = matches_test['match_p_src']

# Build student panorama

tt = tic()
img_pan = panorama(img_src_test, img_dst_test, match_p_src, match_p_dst, inliers_percent, max_err)
print('Student Panorama {:5.4f} sec'.format(toc(tt)))

plt.figure()
panplot = plt.imshow(img_pan)
plt.title('Awesome Panorama')
plt.show()
