import numpy as np
from numpy import random
import ast
import os
import torch
from IPython.display import Image, clear_output  # to display images
from models.experimental import attempt_load
from utils.general import check_img_size, non_max_suppression, scale_coords, xyxy2xywh, increment_path
from utils.torch_utils import select_device
from utils.datasets import LoadImages
from utils.plots import plot_one_box
from pathlib import Path
import cv2
import time
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image
from detect import detect


def run(myAnnFileName, buses):
    with torch.no_grad():
        source = buses
        detection_weights = os.path.join(os.getcwd(), 'detection_weights.pt')
        classifier_weights = os.path.join(os.getcwd(), 'classifier_weights.pth')
        detect( source=source,
                detection_weights=detection_weights,
                imgsz=1824,
                classifier_weights=classifier_weights,
                myAnnFileName=myAnnFileName)
