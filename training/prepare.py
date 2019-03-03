import cv2
import glob
import numpy as np
import os

desired_width = 455
desired_height = 700
input_folder = './dataset_ori'
output_folder = './dataset'

def image_resize(imageFolder, i, height, width):
    image = cv2.imread(imageFolder)
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # first we resize the whole image to the desired width using the right ratio
    r = width / float(w)
    dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)

    # the resized image heigth and width
    (h, w) = resized.shape[:2]

    # short images need to match the desired heigth
    if h < height:
        BLACK = [0, 0, 0]
        resized = cv2.copyMakeBorder(resized, 0 , height - h, 0, 0, cv2.BORDER_CONSTANT, value=BLACK)
    else: 
        resized = resized[0:0+height, 0:0+width]

    # save the resized image
    cv2.imwrite( output_folder + '/' + os.path.basename(imageFolder), resized)

for (i, image_file) in enumerate(glob.iglob(input_folder + '/*.png')):
        image_resize(image_file, i,desired_height, desired_width)
