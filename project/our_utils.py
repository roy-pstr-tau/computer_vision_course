import numpy as np

def crop(img, x_c, y_c, w, h):
    im_h, im_w, _ = img.shape
    x_c = int(x_c * im_w)
    y_c = int(y_c * im_h)
    w = int(w * im_w)
    h = int(h * im_h)
    x_min = x_c - int(w/2)
    y_min = y_c - int(h/2)
    x_slices = slice(x_min, x_min+w)
    y_slices = slice(y_min, y_min+h)
    return img[y_slices,x_slices]

def yolo_to_ours_label(x, w, h):
  label_w = x[2]*w
  label_h = x[3]*h
  x_min = x[0]*w-label_w//2
  y_min = x[1]*h-label_h//2
  return [x_min, y_min, label_w, label_h]
