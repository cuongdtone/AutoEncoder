import cv2
import os
from utils.cfa import Demosaic
from sklearn.model_selection import train_test_split
from glob import glob
import yaml
root_dir = '/'.join(os.path.dirname(__file__).split('/')[:-1])

demosaic = Demosaic()
with open(os.path.join(root_dir, 'src/config.yaml'), 'r') as f:
    param = yaml.load(f, Loader=yaml.FullLoader)
input_size = param['input_size']

def create_cfa_dataset(origin_path, save_path):
    list_classes = glob(origin_path + '/*')
    try:
        os.mkdir(save_path.split('/')[0])
    except:
        pass
    try:
        os.mkdir(save_path)
    except:
        pass
    print('Starting ..')
    for clss in list_classes:
        print('Class: ', clss)
        list_image = glob(clss + '/*.*')
        try:
            os.mkdir(os.path.join(save_path, clss.split('/')[-1]))
        except:
            pass
        for i in list_image:
            try:
                image = cv2.imread(i)
                path_clss_img = ''.join(i.split('/')[-2] + '/' + i.split('/')[-1])
                CFA = demosaic.bgr2cfa(image)
                cv2.imwrite(os.path.join(save_path, path_clss_img), CFA.astype('uint8'))
            except:
                continue
def create_gray_dataset(origin_path, save_path):
    list_classes = glob(origin_path + '/*')
    try:
        os.mkdir(save_path.split('/')[0])
    except:
        pass
    try:
        os.mkdir(save_path)
    except:
        pass
    for clss in list_classes:
        list_image = glob(clss + '/*.*')
        try:
            os.mkdir(os.path.join(save_path, clss.split('/')[-1]))
        except:
            pass
        for i in list_image:
            try:
                image = cv2.imread(i, 0)
                image = cv2.resize(image, (input_size, input_size))
                path_clss_img = ''.join(i.split('/')[-2] + '/' + i.split('/')[-1])
                cv2.imwrite(os.path.join(save_path, path_clss_img), image)
            except:
                continue
def split_dataset(dataset_path, out_path):
    try:
        os.mkdir(out_path)
        os.mkdir(out_path + '/train')
        os.mkdir(out_path + '/test')
    except:
        pass

    list_img = glob(dataset_path+'/*/*')
    list_class = glob(dataset_path+'/*')

    for i in list_class:
        try:
            os.mkdir(out_path+'/train/' + i.split('/')[-1])
            os.mkdir(out_path+'/test/' + i.split('/')[-1])
        except:
            pass
    train, test = train_test_split(list_img)
    for i in train:
        ds_path = out_path + '/train/' + '/'.join(i.split('/')[-2:])
        #print(ds_path)
        cv2.imwrite(ds_path, cv2.imread(i))
    for i in test:
        ds_path = out_path + '/test/' + '/'.join(i.split('/')[-2:])
        cv2.imwrite(ds_path, cv2.imread(i))
if __name__ == '__main__':
    ori = '/home/cuong/Desktop/autoencoder/auto-encoder/dataset_flower_origin (copy)/test'
    ds = ori + '_split'
    split_dataset(ori, ds)