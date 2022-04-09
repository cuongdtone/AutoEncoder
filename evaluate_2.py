import torch
from torch import nn
from models.model import AE_NET, image_torch
import cv2
import glob
import matplotlib.pyplot as plt
import yaml
import random
from utils.plot import plot_cm
from sklearn import metrics

with open('config.yaml', 'r') as f:
    param = yaml.load(f, Loader=yaml.FullLoader)

input_size = param['input_size']
code_size = param['code_size']

model = AE_NET(input_size=input_size, code_size=code_size, feature_etractor='runs/ae.pt')
model.load_state_dict(torch.load('runs/classifier.h5', map_location='cpu'))
model.eval()

print("#Parameter: ", sum(p.numel() for p in model.parameters()))

list_image = glob.glob('dataset_flower_cfa/test/*/*.png')
random.shuffle(list_image)

with open('runs/label.txt', 'r') as f:
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
    image = cv2.imread(i)
    #image[image<=127] = 127
    x = model.feature_extractor.preprocess_image(image)
    out = model(x)
    _, index = torch.max(out, 1)
    percentage = (nn.functional.softmax(out, dim=1)[0] * 100).tolist()
    #print('Predict: ', class_name[index.tolist()[0]], ': %.2f %%'%(max(percentage)))
    truth_clss = i.split('/')[-2]
    #print('Truth: ', truth_clss)
    truth_idx = class_idx[truth_clss]

    pred_label.append(index.tolist()[0])
    truth_label.append(truth_idx)

    # cv2.imshow('ori', image)
    # cv2.waitKey()
# print(pred_label)
# print(truth_label)
CM = metrics.confusion_matrix(truth_label, pred_label)
acc = metrics.accuracy_score(truth_label, pred_label)
print('Accuracy : ', acc)
print([i for i in class_name.values()])
plot_cm(CM, save_dir='runs', names=[i for i in class_name.values()], normalize=False, show=True)