"""
e.g:
python to_yolo.py .\toybus\annotations_train.txt .\toybus\single_class\labels\train\ .\toybus\images\train\ --single_class

YOLO v5 requires the dataset to be in the darknet format. Here’s an outline of what it looks like:

- One txt with labels file per image
- One row per object
- Each row contains: class_index bbox_x_center bbox_y_center bbox_width bbox_height
- Box coordinates must be normalized between 0 and 1

Our given bus dataset format:
- One annotation.txt file for all dataset.
- Each line in text file contains all objects per one image
- e.g. : # PIC.JPG:[xmin1,ymin1,width1,height1,color1],..,[xminN,yminN,widthN,heightN,colorN]

Out classes are: {green_bus=1, yellow_bus=2, white_bus=3, gray_bus=4, blue_bus=5, red_bus=6} 

"""

import click
import ast
from pathlib import Path
import cv2

@click.command()
@click.argument('annot_path', type=str)
@click.argument('labels_folder', type=str)
@click.argument('images_folder', type=str)
@click.option('--single_class', is_flag=True, default=False)
@click.option('--verbose', is_flag=True, default=False)
def to_yolo(annot_path, labels_folder, images_folder, single_class=False, verbose=False):
    with open(annot_path, 'r') as annot_file:
        annot_lines = annot_file.readlines()
        for image_line in annot_lines:
            image_name = image_line.split('.')[0]
            objects_in_im = image_line.split(':')[1]
            objects_in_im_lst = ast.literal_eval(str('['+objects_in_im+']')) 
            label_file_name = Path(labels_folder) / str(image_name +".txt")
            im = cv2.imread(str(Path(images_folder)/ str(image_name+'.jpg')))
            im_h, im_w, im_c = im.shape
            click.secho(f"image: {image_name} ({im_w},{im_h}), objects: {objects_in_im_lst}\nobjects:", fg="blue")
            class_type = 0
            with open(label_file_name, 'w') as label_file:
                for obj in objects_in_im_lst:
                    xmin, ymin, width, height, color = obj
                    xcenter = (xmin + (width/2))
                    ycenter = (ymin + (height/2))
                    if not single_class:
                        class_type = color-1 
                    label_file.write(str(class_type) + ' ' + str(xcenter / im_w) + ' ' + str(ycenter / im_h) + ' ' + str(width / im_w) + ' ' + str(height / im_h) + '\n')
                    click.secho(f"\t {str(class_type)} {str(xcenter / im_w)} {str(ycenter / im_h)} {str(width / im_w)} {str(height / im_h)}", fg="green")

        
def from_yolo():
    pass

if __name__=='__main__':
    to_yolo()

 