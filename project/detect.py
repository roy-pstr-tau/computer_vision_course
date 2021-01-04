from pathlib import Path
from numpy import random
import cv2
import time
import matplotlib.pyplot as plt
import numpy as np
from torchvision import  transforms
from PIL import Image
import torch
from models.experimental import attempt_load
from utils.general import check_img_size, non_max_suppression, scale_coords, xyxy2xywh, increment_path
from utils.torch_utils import select_device
from utils.datasets import LoadImages
from utils.plots import plot_one_box
from our_utils import crop, yolo_to_ours_label

test_transforms = transforms.Compose([transforms.Resize(224), transforms.ToTensor()])
def color_classification(img, xywh, model, device, save_path):
    cropped_img = crop(img, *xywh)
    #print(cropped_img.shape)
    #img = transforms.functional.to_pil_image(cropped_img) # this one does not works well...
    cv2.imwrite( save_path, cropped_img) # we need to figure a way to skip the save and load....
    img = Image.open(save_path) 
    img = test_transforms(img)
    if img.ndimension() == 3:
        img = img.unsqueeze_(0)
    return model(img.to(device).half()).argmax(1)

def detect(source, detection_weights, imgsz, classifier_weights, myAnnFileName):
    t0 = time.time()
    save_txt = True
    save_img = True
    save_crops = True
    view_img = False

    # Directories
    save_dir = Path(increment_path(Path("detected") / "exp", exist_ok=False))  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir
    (save_dir / 'cropped' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir
    device = select_device() 
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load bus detector model:
    model = attempt_load(detection_weights, map_location=device)  # load FP32 model
    imgsz = check_img_size(imgsz, s=model.stride.max())  # check img_size TODO check this limitation!
    

    # Load color classifier:
    cmodel=torch.load(classifier_weights)
    cmodel.eval()
    
    if half:
        model.half()  # to FP16
        cmodel.half()

    dataset = LoadImages(source, img_size=imgsz)
    # Get names and colors
    classes = ['blue_bus', 'gray_bus', 'green_bus', 'red_bus', 'white_bus', 'yellow_bus']
    #theirs_classes = ['green_bus', 'yellow_bus', 'white_bus', 'gray_bus', 'blue_bus', 'red_bus']
    ours_theirs_mapping = [4,3,0,5,2,1]
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in classes]

     # Create Annotation File
    f = open(myAnnFileName, "w+")

    for path, img, img_original, _ in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        pred = model(img)[0]
        #print(pred.shape)
        # Apply NMS
        pred = non_max_suppression(pred, conf_thres=0.25, iou_thres=0.45)
        #print(pred[0].shape)
        # Process detections
        output_image = np.copy(img_original)
        # write image name in Annotation File
        f.write(Path(path).name + ":")
        for i, det in enumerate(pred):  # detections per image
            p, s, im0 = Path(path), '', img_original
            #print(i,det)
            save_path = str(save_dir / p.name)
            txt_path = str(save_dir / 'labels' / p.stem)
            s += '%gx%g ' % img.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += '%g %ss, ' % (n, names[int(c)])  # add to string

                # Write results
                for j, (*xyxy, conf, cls) in enumerate(reversed(det)):
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        color_cls = color_classification(img=im0, xywh=xywh, model=cmodel, device=device, save_path=str(save_dir / "cropped"/f"{p.stem}_{j}.jpg"))
                        print(classes[color_cls])
                        line = (color_cls, *xywh)  # label format
                        # with open(txt_path + '.txt', 'a') as f:
                        #     f.write(('%g ' * len(line)).rstrip() % line + '\n')
                        our_xywh = yolo_to_ours_label(xywh, im0.shape[1],im0.shape[0])
                        #print(f"{xywh} -> {our_xywh}, [{im0.shape[1]},{im0.shape[0]}]")
                        our_xywh.append((float(ours_theirs_mapping[color_cls])+1))
                        if j == len(reversed(det))-1:
                            f.write(str([int(i) for i in our_xywh]).replace(" ", ""))
                            #print(str([int(i) for i in our_xywh]).replace(" ", ""))
                        else:
                            f.write(str([int(i) for i in our_xywh]).replace(" ", "") + ",")
                            #print(str([int(i) for i in our_xywh]).replace(" ", "") + ",")
                    if save_img or view_img:  # Add bbox to image
                        label = '%s %.2f' % (classes[color_cls], conf)
                        plot_one_box(xyxy, output_image, label=label, color=colors[int(color_cls)], line_thickness=3)

            # Stream results
            if view_img:
                cv2.imshow(str(p), output_image)
                if cv2.waitKey(1) == ord('q'):  # q to quit
                    raise StopIteration

            # Save results (image with detections)
            if save_img:
                cv2.imwrite(save_path, output_image)

        f.write("\n")

    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        print(f"Results saved to {save_dir}{s}")
    f.close()
    print('Done. (%.3fs)' % (time.time() - t0))