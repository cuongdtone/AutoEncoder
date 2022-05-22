import cv2
import random
from scipy import ndimage
import os
from glob import glob
import shutil

def brightness(image, value_min, value_max):
    """
    Function: change brightness of a image
    :param image: image original
    :param value_min: value min
    :param value_max: value max
    :return: new image
    """
    # convert bgr - > hsv
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # split hsv
    h, s, v = cv2.split(image)
    # random value light
    value = random.randint(value_min, value_max)
    # check and increase, decrease light
    if value > 0:
        lim = 255 - value
        v[v > lim] = 255
        v[v <= lim] += value
    else:
        lim = -value
        v[v < lim] = 0
        v[v >= lim] -= lim
    # merger hsv
    image_hsv = cv2.merge([h, s, v])
    # convert hsv ->  bgr
    result = cv2.cvtColor(image_hsv, cv2.COLOR_HSV2BGR)
    return result


def rotate(image):
    """
    Function: Rotate image with three angle
    :param image: image original
    :return: image rotate
    """
    # random angle
    list_angle = [90, 180, 270]
    idx = random.randint(0, 2)
    # rotate
    result = ndimage.rotate(image, list_angle[idx])
    return result


def sharpening(image, sigma=1.0):
    """
    Function: The shaperning for a image
    :param image: image original
    :param sigma: value sigma
    :return: image sharpening
    """
    # random amount
    amount = random.randint(1, 3)
    gauss = cv2.GaussianBlur(image, (3, 3), sigma)
    # add image original and gauss
    result = cv2.addWeighted(image, 1 + amount, gauss, -amount, 0)
    return result


def agumentation(path_dataset):
    """
    Function: Augumentation for folder image
    :param path_dataset: path contain data
    :return: folder new contain image augumentation
    """
    print('Processing augumentation...')
    # create folder save and get path name folder flowers
    list_data = os.path.join(path_dataset, 'train')
    path_new = path_dataset + '_agumentation'
    if not os.path.exists(path_new):
        os.mkdir(path_new)

    list_name = glob(list_data + '/*')
    # get name folder and create folder in folder data new
    list_class = [i.split('/')[-1] for i in list_name]
    for name_class in list_class:
        os.mkdir(os.path.join(path_new, name_class))
    # get image and augumentation
    for path_image in list_name:
        list_image = glob(path_image + '/*.png')
        for path in list_image:
            image = cv2.imread(path)

            # get name folder
            name_flower = path_image.split('/')[-1]
            shutil.copy(path,  f'{path_new}/{name_flower}/')
            # get name image
            name_image = path.split('/')[-1].split('.')[0]
            # augumentation
            for i in [-1, 0]:
                # flip vertical and horizontal
                result_flip = cv2.flip(image, i)
                # change brightness
                result_brightness = brightness(result_flip, -30, 30)
                # rotate image
                result_rotate = rotate(result_brightness)
                # shapening image
                result_sharp = sharpening(result_rotate)
                # save image
                path_save = f'{path_new}/{name_flower}/{name_image}_{abs(i)}.jpg'
                cv2.imwrite(path_save, result_sharp)
    print('Finish.')


if __name__ == '__main__':
    agumentation('dataset_flower_origin')




