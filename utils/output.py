import os
import cv2

def save_new_image(path_dir, image):
    c = 0
    while (os.path.exists(os.path.join(path_dir, 'test_%d.jpg'%(c)))):
        c = c + 1
    path = os.path.join(path_dir, 'test_%d.jpg'%(c))
    cv2.imwrite(path, image)