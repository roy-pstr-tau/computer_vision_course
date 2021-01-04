import click
import os
from pathlib import Path
import imageio

classes = ["green", "yellow", "white", "gray", "blue", "red"]
@click.command()
@click.argument('images', type=str)
@click.argument('labels', type=str)
@click.argument('save_to', type=str)
def run(images, labels, save_to): 
    for filename in os.listdir(images):
        if filename.endswith(".jpg") or filename.endswith(".JPG"): 
            image_file_path = Path(images) / filename
            image_name = image_file_path.stem
            label_file_path = Path(labels) / str(image_name+".txt")
            labels_in_image = load_lables(label_file_path)
            if len(labels_in_image)>0:
                im=imageio.imread(image_file_path, pilmode="RGB") 
                crop_and_save(image_name,im, labels_in_image, save_to)

def load_lables(labels_path):
    with open(labels_path, 'r') as f:
        return [line.split() for line in f.readlines()]

def crop_and_save(image_name, im, labels, save_to):
    classes_counts = dict.fromkeys(classes, 0) 
    for label in labels:
        label = [float(num) for num in label] 
        cropped_im = crop(im, label[1], label[2], label[3], label[4])
        class_name = classes[int(label[0])]
        curr_save_to = Path(save_to) / f"{class_name}_bus" / f"{image_name}_{classes_counts[class_name]}.jpg"
        imageio.imwrite(curr_save_to, cropped_im)
        classes_counts[class_name] += 1
        click.secho(str(curr_save_to), fg="green")

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

if __name__=="__main__":
    run()