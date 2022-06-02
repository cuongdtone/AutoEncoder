import torch
from torch import nn
from models.model import AE, Net, image_torch
import cv2
import glob
import matplotlib.pyplot as plt
import yaml
import random
from utils.plot import plot_cm
from sklearn import metrics

with open('src/config.yaml', 'r') as f:
    param = yaml.load(f, Loader=yaml.FullLoader)

input_size = param['input_size']
code_size = param['code_size']

feature_extractor = AE(input_size=input_size, code_size=param['code_size'])
feature_extractor.load_state_dict(torch.load('runs/ae.pt', map_location='cpu'))
feature_extractor.eval()

net = Net(code_size)
net.load_state_dict(torch.load('runs/classifier.pt', map_location='cpu'))
net.eval()

list_image = glob.glob('dataset_flower_cfa/test/*/*.png')
random.shuffle(list_image)

with open('../runs/label.txt', 'r') as f:
    class_name = f.readlines()
    class_name = {int(i.split(':')[1].strip('\n')) : i.split(':')[0].strip() for i in class_name}
    class_idx = dict((v, k) for k, v in class_name.items())

print(class_name)
print(class_idx)
# cv2.namedWindow('ori', cv2.WINDOW_NORMAL)
# cv2.resizeWindow('ori', 480, 480)

truth_label = []
pred_label = []
for i in list_image:
    image = cv2.imread(i, 0)
    #image[image<=127] = 127
    x = feature_extractor.preprocess_image(image)
    code = feature_extractor.get_coding(x)
    out = net(code)

    _, index = torch.max(out, 1)
    percentage = (nn.functional.softmax(out, dim=1)[0] * 100).tolist()
    truth_clss = i.split('/')[-2]
    # print('Predict: ', class_name[index.tolist()[0]], ': %.2f %%'%(max(percentage)))
    # print('Truth: ', truth_clss)
    truth_idx = class_idx[truth_clss]

    pred_label.append(index.tolist()[0])
    truth_label.append(truth_idx)

    # cv2.imshow('ori', image)
    # cv2.waitKey()
# print(pred_label)
# print(truth_label)
CM = metrics.confusion_matrix(pred_label, truth_label)
acc = metrics.accuracy_score(pred_label, truth_label)
print('Accuracy : %d %%'%(acc*100))
plot_cm(CM, save_dir='../runs', names=[i for i in class_name.values()], normalize=False, show=True)