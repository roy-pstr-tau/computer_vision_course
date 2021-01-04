import os
from detect import detect
import torch

def run(myAnnFileName, buses):
    with torch.no_grad():
        source = buses
        detection_weights = os.path.join(os.getcwd(), 'weights/detection_weights.pt')
        classifier_weights = os.path.join(os.getcwd(), 'weights/classifier_weights.pth')
        detect( source=source,
                detection_weights=detection_weights,
                imgsz=1824,
                classifier_weights=classifier_weights,
                myAnnFileName=myAnnFileName)
