import os
from pathlib import Path
import imageio
import numpy as np
import matplotlib.pyplot as plt
colors = ['blue', 'gray', 'green', 'red', 'white', 'yellow']
#colors = ['blue']
folders = [f"toybus\cropped_bus\\{color}_bus" for color in colors]

def load_images(folder):
    images = []
    for filename in os.listdir(folder):
            if filename.endswith(".jpg") or filename.endswith(".JPG"): 
                image_file_path = Path(folder) / filename
                im=imageio.imread(image_file_path, pilmode="RGB") 
                images.append(im)
    print(f"Loaded {len(images)} images.")
    return images

def channels_average(pic):
    return np.mean(pic[::,::,0]), np.mean(pic[::,::,1]), np.mean(pic[::,::,2])

def center_crop(images, w_ratio, h_ratio):
    cropped = []
    for img in images:
        h,w,_ = img.shape
        center_x = int(w/2)
        center_y = int(h/2)
        delta_x = int((w_ratio * w) / 2)
        delta_y = int((h_ratio * h) / 2)
        x_slices = slice(center_x - delta_x, center_x + delta_x)
        y_slices = slice(center_y - delta_y, center_y + delta_y)
        croped_img = img[y_slices,x_slices]
        cropped.append(croped_img)
    return cropped

buses_by_color = {}
for i, folder in enumerate(folders):
    buses = load_images(folder)
    color = colors[i]
    buses_by_color[color] = buses

colors_avg = {}
centered_cropped_buses_by_color = {}
for color, buses in buses_by_color.items():
    rgb_means = np.array([])
    centered_cropped_buses_by_color[color] = center_crop(buses, 0.5, 0.5)
    for bus in centered_cropped_buses_by_color[color]:
        red_mean, green_mean, blue_mean = channels_average(bus)
        curr = np.array([red_mean, green_mean, blue_mean])
        if rgb_means.size!=0:
            rgb_means = np.vstack([rgb_means,curr])
        else:
            rgb_means = curr
    colors_avg[color] = rgb_means.mean(axis=0)

# colors_avg = {
# "blue": [104.46847761, 114.13294888 ,120.37792454],
# "gray": [119.3669766 , 108.04864623 ,103.20000898],
# "green": [ 83.55157513 ,113.59169103  ,80.61566398],
# "red": [148.73039551  ,81.81706835  ,82.14515119],
# "white": [140.806023   ,125.9447623  ,124.27810646],
# "yellow": [161.70926096 ,119.10544839  ,80.95815193]
# }
accuracy = {}
for color, buses in centered_cropped_buses_by_color.items():
    rgb_means = np.array([])
    accuracy[color] = [0,0]
    for bus in buses:
        red_mean, green_mean, blue_mean = channels_average(bus)
        img_means = np.array([red_mean, green_mean, blue_mean])
        min_mse = np.array([255, 255, 255])
        color_detected = None
        accuracy[color][0] += 1
        for c, means in colors_avg.items():
            rgb_means = np.array(means)
            mse = np.abs(rgb_means - img_means)
            if (mse < min_mse).all():
                min_mse = mse
                color_detected = c
            if c==color:
                correct_mse=mse
        if color==color_detected:
            accuracy[color][1] += 1
        # print(f"({color},{color_detected}) ({color==color_detected}), {min_mse} , {correct_mse}")
        print(f"({color},{color_detected}) ({color==color_detected})")

print(accuracy)