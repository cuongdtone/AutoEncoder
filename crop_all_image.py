import numpy as np
from glob import glob
import cv2
import os

def crop_binary_image(I):
    [rows, cols] = np.where(I)
    min_col = min(cols)
    max_col = max(cols)
    min_row = min(rows)
    max_row = max(rows)

    idx = [min_row,max_row,min_col,max_col]
    J = I[idx[0]:idx[1], idx[2]:idx[3]]

    return J,idx

def crop_rgb_image(J,idx):
    img = J[idx[0]:idx[1],idx[2]:idx[3],:]
    return img


if __name__ == '__main__':
    #input
    Binary_input = '../datasets/binary_mask'
    Original_input = '../datasets/original'

    #output
    Binary_crop = '../datasets/binary_crop'
    Original_crop ='../datasets/original_crop'



    ##
    classes = ['rose','daisy','sunflower','lotus','hibiscus']

    for cls in classes:

        binary_input = os.path.join(Binary_input,cls)
        original_input = os.path.join(Original_input,cls)
        binary_crop = os.path.join(Binary_crop,cls)
        original_crop = os.path.join(Original_crop ,cls)

        try:
            os.makedirs(binary_crop,exist_ok = True)
            os.makedirs(original_crop,exist_ok = True)
        except:
            print("Directory can not be created" )


        list_images = glob(binary_input  + '/*.png')
        total_imges = len(list_images)

        print("{} : {} images".format(cls,total_imges))
        for binary_input_path in list_images:

            original_input_path = binary_input_path .replace(binary_input,original_input)
            binary_output_path = binary_input_path.replace(binary_input, binary_crop)
            original_output_path = original_input_path .replace(original_input, original_crop)

            I = cv2.imread(binary_input_path,0) #binary image
            J = cv2.imread(original_input_path) #original image

            I = 255 * (I > 0)
            I = I.astype(np.uint8)
            
            try:
                I,idx = crop_binary_image(I)        #crop binary and return idx
                J = crop_rgb_image(J,idx)           #crop rbg image with idx from mask

                cv2.imwrite(binary_output_path,I)
                cv2.imwrite(original_output_path, J)
            except:
                print(binary_input_path,'error !'








