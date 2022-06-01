# -*- coding: utf-8 -*-
# @Organization  : DUT
# @Author        : Cuong Tran
# @Time          : 01/06/2022


import cv2
from colour_demosaicing import demosaicing_CFA_Bayer_Menon2007, mosaicing_CFA_Bayer, demosaicing_CFA_Bayer_bilinear
import yaml
import os
root_dir = '/'.join(os.path.dirname(__file__).split('/')[:-1])

'''Input and output is opencv format'''

class Demosaic():
    def __init__(self):
        with open(os.path.join(root_dir, 'src/config.yaml'), 'r') as f:
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

demosaic = Demosaic()
