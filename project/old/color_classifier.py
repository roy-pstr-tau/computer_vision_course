import imageio
import matplotlib.pyplot as plt
import cv2
import numpy as np
import scipy.misc

def crop(imgs, x_c, y_c, w, h):
    im_h, im_w, _ = im.shape
    x_c = int(x_c * im_w)
    y_c = int(y_c * im_h)
    w = int(w * im_w)
    h = int(h * im_h)
    x_min = x_c - int(w/2)
    y_min = y_c - int(h/2)
    x_slices = slice(x_min, x_min+w)
    y_slices = slice(y_min, y_min+h)
    return (imgs[0][y_slices,x_slices], imgs[1][y_slices,x_slices])

def channels_average(pic):
    return np.mean(pic[::,::,0]), np.mean(pic[::,::,1]), np.mean(pic[::,::,2])

def rgb_hist(pic, ax):
    color = ('b','g','r')
    for i,col in enumerate(color):
        histr = cv2.calcHist([pic],[i],None,[256],[0,256])
        ax.plot(histr,color = col)
        #ax.xlim([0,256])

def hsv_hist(pic, ax):
    h, s, v = cv2.split(pic)
    h_histr = cv2.calcHist(h,[0],None,[180],[0,179])
    print(h_histr.argmax(), max(h_histr))
    ax.plot(h_histr)


im=imageio.imread("toybus/images/train/DSCF1013.JPG", pilmode="RGB") 
im_hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
## convert to hsvb

buses = [crop((im,im_hsv), 0.40063048245614036, 0.654422514619883, 0.13404605263157895, 0.07346491228070176),
        crop((im,im_hsv), 0.5513980263157895, 0.6326754385964912, 0.13020833333333334, 0.08187134502923976),
        crop((im,im_hsv), 0.6970942982456141, 0.6140350877192983, 0.12609649122807018, 0.08333333333333333),
        crop((im,im_hsv), 0.42009320175438597, 0.7118055555555556, 0.13623903508771928, 0.08442982456140351),
        crop((im,im_hsv), 0.5816885964912281, 0.6988304093567251, 0.13322368421052633, 0.08333333333333333),
        crop((im,im_hsv), 0.7309484649122807, 0.6783625730994152, 0.13020833333333334, 0.08333333333333333)]

def detect_color(pic):
    colors = [
	("blue",[50,65,100]),
	("red",[175,10,20]),
    ("green",[10,130,60]),
	("yellow",[215,130,55]),
	("grey",[170,130,120]),
    ("white",[215,215,210])
    ]    
    pixels_count = []
    # loop over the boundaries
    thres = 10
    for (name,rgb) in colors:

        # create NumPy arrays from the boundaries
        lower = np.array(rgb, dtype="uint8")-thres
        upper = np.array(rgb, dtype="uint8")+thres
        count = pic[np.where((pic>lower) & (pic<upper))].shape[0]
        pixels_count.append((name,count))
    
    max_count = 0
    i_max=-1
    for i,(color, count) in enumerate(pixels_count):
        if count>max_count: 
            max_count=count
            i_max=i
    print(pixels_count, pixels_count[i_max])
        
for i in range(6):
    detect_color(buses[i][0])

fig, axs = plt.subplots(nrows=4, ncols=3)
for i,bus in enumerate(buses):
    red_mean, green_mean, blue_mean = channels_average(bus[0])
    col = i % 3
    row = int(i / 3)
    axs[row*2, col].title.set_text(f"blue: {int(blue_mean)} green: {int(green_mean)} red: {int(red_mean)}")
    h, s, v = cv2.split(bus[1])
    axs[row*2, col].imshow(bus[0])
    rgb_hist(bus[0], axs[row*2+1, col])
fig.tight_layout()
plt.show()



# fig, axs = plt.subplots(nrows=1, ncols=3)
# for col, (lower, upper) in enumerate(boundaries):
#     print(lower, upper)
#     mask = cv2.inRange(buses[5], lower, upper)
#     axs[0, col].imshow(bus)
# plt.show()

