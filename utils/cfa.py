from glob import glob
import cv2
import numpy as np
import os
import sys
import colour
from colour_demosaicing import demosaicing_CFA_Bayer_Menon2007, mosaicing_CFA_Bayer, demosaicing_CFA_Bayer_bilinear
import yaml
import os
root_dir = '/'.join(os.path.dirname(__file__).split('/')[:-1])

'''Input and output is opencv format'''

class Demosaic():
    def __init__(self):
        with open(os.path.join(root_dir, 'config.yaml'), 'r') as f:
            param = yaml.load(f, Loader=yaml.FullLoader)
        self.input_size = param['input_size']
    def bgr2cfa(self, image):
        # image is opencv image
        image = cv2.resize(image, (self.input_size, self.input_size))/255
        CFA = mosaicing_CFA_Bayer(image)
        CFA = (CFA * 255).astype('uint8')
        return CFA
    def cfa2bgr(self, cfa):
        cfa = cfa/255
        rgb = demosaicing_CFA_Bayer_bilinear(cfa)
        rgb = rgb*255
        return rgb.astype('uint8')
if __name__ == '__main__':
    image = cv2.imread('/home/cuong/Desktop/autoencoder/auto-encoder/dataset_flower/hibiscus/hibiscus_2.png')
    demosaic = Demosaic()
    CFA = demosaic.bgr2cfa(image)
    BGR = demosaic.cfa2bgr(CFA)
    cv2.imshow('ori', image)
    cv2.imshow('cfa', CFA)
    cv2.imshow('bgr', BGR)
    cv2.waitKey()
