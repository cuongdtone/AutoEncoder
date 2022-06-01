# -*- coding: utf-8 -*-
# @Organization  : DUT
# @Author        : Cuong Tran
# @Time          : 01/06/2022
# @Function      : User Interface

import cv2
import numpy as np
import pyqtgraph as pg
import os
import sys
import traceback
import torch
from torchvision import datasets
from torch import nn, optim
import yaml
import time
import glob
import random
from sklearn import metrics
from pathlib import Path

from PyQt5 import QtGui
from PyQt5.QtWidgets import QMainWindow, QApplication, QWidget, QTabWidget, QVBoxLayout, QMessageBox, QLabel, QFileDialog
from PyQt5.QtGui import QIcon, QPixmap
from PyQt5.QtCore import pyqtSignal, pyqtSlot, Qt, QThread

from models.model import AE, image_torch, Net
from utils.transform import transformer
from utils.cfa import demosaic
from utils.functions import concat_image
from utils.plot import plot_cm
from src.home import Ui_Form as home
from src.test_tab import Ui_Form as infference
from sklearn import svm
import pickle
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TrainSVM(QThread):
    change_infor_signal = pyqtSignal(str)
    
    def __init__(self, dataset_dir):
        super(TrainSVM, self).__init__()
        self.dataset_dir = dataset_dir
        self.batch_size = 1
        
    def run(self):
        input_path = os.path.join(self.dataset_dir, 'train')
        dir_save_model = 'runs'
        with open('src/config.yaml', 'r') as f:
            param = yaml.load(f, Loader=yaml.FullLoader)
        input_size = param['input_size']
        code_size = param['code_size']
        feature_extractor = torch.load('runs/ae.pt', map_location=device)
        feature_extractor.eval()
        dataset = datasets.ImageFolder(input_path, transformer)
        self.change_infor_signal.emit(('Total image: %d' % (len(dataset))))
        train_loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        last_time = time.time()
        list_data = []
        list_label = []
        for batch_features, labels in train_loader:
            batch_features = batch_features.view(-1, input_size ** 2).to(device)
            feature = feature_extractor.get_coding(batch_features)
            feature = feature.cpu().detach().numpy()[0]
            labels = labels.numpy()[0]
            list_data.append(feature)
            list_label.append(labels)

        X = np.array(list_data)
        y = np.array(list_label)

        SVM = svm.SVC(decision_function_shape='ovo')
        SVM.fit(X, y)
        # save the model to disk
        filename = f'{dir_save_model}/svm.pickle'
        pickle.dump(SVM, open(filename, 'wb'))
        self.change_infor_signal.emit('Accuracy: {}, Total time: {}'.format(SVM.score(X, y), time.time()-last_time))
        self.change_infor_signal.emit('Complete ! Model saved at svm.pickle')


class EvaluateSVM(QThread):
    change_screen_signal = pyqtSignal(np.ndarray)

    def __init__(self, dataset_dir):
        super(EvaluateSVM, self).__init__()
        self.dataset_dir = dataset_dir
        self.list_img_train = glob.glob(os.path.join(dataset_dir, 'train/*/*'))
        self.list_img_test = glob.glob(os.path.join(dataset_dir, 'test/*/*'))
        with open('src/config.yaml', 'r') as f:
            param = yaml.load(f, Loader=yaml.FullLoader)
        self.input_size = param['input_size']
        self.code_size = param['code_size']
        self.run_flag = True

    def run(self):
        while self.run_flag:
            try:
                time.sleep(0.1)
                feature_extractor = torch.load('runs/ae.pt', map_location=device)
                feature_extractor.eval()
                svm_classification = pickle.load(open('runs/svm.pickle', 'rb'))
                with open('runs/label.txt', 'r') as f:
                    class_name = f.readlines()
                    class_name = {int(i.split(':')[1].strip('\n')): i.split(':')[0].strip() for i in class_name}
                    class_idx = dict((v, k) for k, v in class_name.items())
                truth_label = []
                pred_label = []
                for i in self.list_img_test:
                    image = cv2.imread(i, 0)
                    x = feature_extractor.preprocess_image(image)
                    code = feature_extractor.get_coding(x)
                    code = code.cpu().detach().numpy()
                    truth_clss = i.split('/')[-2]
                    truth_idx = class_idx[truth_clss]
                    index = svm_classification.predict(code)[0]
                    pred_label.append(index)
                    truth_label.append(truth_idx)
                CM = metrics.confusion_matrix(pred_label, truth_label)
                acc = metrics.accuracy_score(pred_label, truth_label)
                plot_cm(CM, save_dir='runs', names=[i for i in class_name.values()],
                        normalize=False, show=False, title='Acc = %.2f %%'%(acc*100))
                show = cv2.imread('runs/confusion_matrix.png')
                show = cv2.resize(show, (641, 731))
                self.change_screen_signal.emit(show)
            except:
                continue


class TrainNeural(QThread):
    change_infor_signal = pyqtSignal(str)
    change_loss_signal = pyqtSignal(float)
    change_acc_signal = pyqtSignal(float)

    def __init__(self, dataset_dir, epochs=40, batch_size=128, lr=1e-3):
        super(TrainNeural, self).__init__()
        self.run_flag = True
        self.complete_epoch_flag = False
        self.dataset_dir = dataset_dir
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.run_flag = True

    def run(self):
        input_path = os.path.join(self.dataset_dir, 'train')
        dir_save_model = 'runs'
        with open('src/config.yaml', 'r') as f:
            param = yaml.load(f, Loader=yaml.FullLoader)
        input_size = param['input_size']
        code_size = param['code_size']
        feature_extractor = torch.load('runs/ae.pt', map_location=device)
        feature_extractor.eval()
        net = Net(input_size=code_size, num_classes=4)
        optimizer = optim.Adam(net.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss()
        dataset = datasets.ImageFolder(input_path, transformer)
        self.change_infor_signal.emit(('Total image: %d'%(len(dataset))))
        file = open('runs/label.txt', "w")
        class_id = dataset.class_to_idx
        for key, value in class_id.items():
            file.write(key + " : " + str(value) + '\n')
        file.close()
        train_loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        for epoch in range(self.epochs):
            loss = 0
            running_corrects = 0
            last_time = time.time()
            for batch_features, labels in train_loader:
                batch_features = batch_features.view(-1, input_size ** 2).to(device)
                feature = feature_extractor.get_coding(batch_features)
                outputs = net(feature)
                train_loss = criterion(outputs, labels)
                optimizer.zero_grad()
                train_loss.backward()
                optimizer.step()
                loss += train_loss.item()
                _, preds = torch.max(outputs, 1)
                running_corrects += torch.sum(preds == labels.data)
                if not self.run_flag:
                    self.change_infor_signal.emit("Stop !")
                    return
            loss = loss / len(train_loader)
            epoch_acc = running_corrects.double() / len(dataset)
            total_memory, used_memory_after, free_memory = map(int, os.popen('free -t -m').readlines()[-1].split()[1:])
            self.change_loss_signal.emit(loss)
            self.change_acc_signal.emit(epoch_acc)
            self.change_infor_signal.emit("Epoch : {}/{}, loss = {:.4f}, acc = {:.2f}, time = {:.2f}, Mem = {:d}/{:d} MB".format(epoch + 1,
                                                                                              self.epochs,
                                                                                              loss,
                                                                                              epoch_acc,
                                                                                              time.time() - last_time,
                                                                                              used_memory_after,
                                                                                              total_memory))
            torch.save(net, os.path.join(dir_save_model, 'classifier.pt'))
        self.change_infor_signal.emit('Complete ! Model saved at classifier.pt')


class EvaluateNeural(QThread):
    change_screen_signal = pyqtSignal(np.ndarray)

    def __init__(self, dataset_dir):
        super(EvaluateNeural, self).__init__()
        self.dataset_dir = dataset_dir
        self.list_img_train = glob.glob(os.path.join(dataset_dir, 'train/*/*'))
        self.list_img_test = glob.glob(os.path.join(dataset_dir, 'test/*/*'))
        with open('src/config.yaml', 'r') as f:
            param = yaml.load(f, Loader=yaml.FullLoader)
        self.input_size = param['input_size']
        self.code_size = param['code_size']
        self.run_flag = True

    def run(self):
        while self.run_flag:
            try:
                time.sleep(0.1)
                feature_extractor = torch.load('runs/ae.pt', map_location=device)
                feature_extractor.eval()
                classifier = torch.load('runs/classifier.pt', map_location=device)
                classifier.eval()

                with open('runs/label.txt', 'r') as f:
                    class_name = f.readlines()
                    class_name = {int(i.split(':')[1].strip('\n')): i.split(':')[0].strip() for i in class_name}
                    class_idx = dict((v, k) for k, v in class_name.items())
                truth_label = []
                pred_label = []
                for i in self.list_img_train:
                    image = cv2.imread(i, 0)
                    x = feature_extractor.preprocess_image(image)
                    code = feature_extractor.get_coding(x)
                    out = classifier(code)
                    _, index = torch.max(out, 1)
                    truth_clss = i.split('/')[-2]
                    truth_idx = class_idx[truth_clss]
                    pred_label.append(index.tolist()[0])
                    truth_label.append(truth_idx)
                CM = metrics.confusion_matrix(pred_label, truth_label)
                acc = metrics.accuracy_score(pred_label, truth_label)
                plot_cm(CM, save_dir='runs', names=[i for i in class_name.values()],
                        normalize=False, show=False, title='Acc = %.2f %%'%(acc*100))
                show = cv2.imread('runs/confusion_matrix.png')
                show = cv2.resize(show, (641, 731))
                self.change_screen_signal.emit(show)
            except:
                continue


class TrainAutoencoder(QThread):
    change_infor_signal = pyqtSignal(str)
    change_loss_signal = pyqtSignal(float)
    change_time_signal = pyqtSignal(float)

    def __init__(self, dataset_dir='', epochs=200, batch_size=128, lr=1e-3):
        super(TrainAutoencoder, self).__init__()
        self.run_flag = True
        self.complete_epoch_flag = False
        self.dataset_dir = dataset_dir
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr

    def run(self):
        input_path = os.path.join(self.dataset_dir, 'train')
        dir_save_model = 'runs'
        with open('src/config.yaml', 'r') as f:
            param = yaml.load(f, Loader=yaml.FullLoader)
        input_size = param['input_size']
        code_size = param['code_size']
        model = AE(input_size=input_size, code_size=code_size).to(device)
        model = torch.load(dir_save_model + '/ae.pt') #use checkpoint
        self.change_infor_signal.emit("#Parameter: %d"%(sum(p.numel() for p in model.parameters())))
        optimizer = optim.Adam(model.parameters(), lr=self.lr)
        criterion = nn.MSELoss()
        preprocess = transformer
        dataset = datasets.ImageFolder(input_path, preprocess)
        self.change_infor_signal.emit("Total images: %d"%(len(dataset)))
        train_loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        self.change_infor_signal.emit("Start traning ...")
        loss_hist = []
        for epoch in range(self.epochs):
            self.complete_epoch_flag = False
            last_time = time.time()
            loss = 0
            for batch_features, _ in train_loader:
                try:
                    batch_features = batch_features.view(-1, input_size ** 2).to(device)  # flatten
                    optimizer.zero_grad()  # grad = 0
                    outputs = model(batch_features)
                    train_loss = criterion(outputs, batch_features)
                    train_loss.backward()
                    optimizer.step()
                    loss += train_loss.item()
                    if not self.run_flag:
                        self.change_infor_signal.emit("Stop !")
                        return
                except:
                    continue
            loss = loss / len(train_loader)
            loss_hist.append(loss)
            total_memory, used_memory_after, free_memory = map(int, os.popen('free -t -m').readlines()[-1].split()[1:])
            self.change_loss_signal.emit(loss)
            self.change_infor_signal.emit("Epoch: {}/{}, loss = {:.4f}, time = {:.1f}, Memory = {:d}/{:d} MB".format(epoch + 1, self.epochs, loss,
                                                                                              time.time() - last_time,
                                                                                              used_memory_after,
                                                                                              total_memory))
            torch.save(model, os.path.join(dir_save_model, 'ae.pt'))
            self.change_time_signal.emit(time.time() - last_time)
            self.complete_epoch_flag = True
        self.change_infor_signal.emit('Completed !')


class EvaluateAutoencoder(QThread):
    change_screen_signal = pyqtSignal(np.ndarray)
    def __init__(self, dataset_dir, time_per_epoch=7, cfa=True):
        super(EvaluateAutoencoder, self).__init__()
        self.dataset_dir = dataset_dir
        self.cfa_flag = cfa
        self.run_flag = True
        self.time_per_epoch = time_per_epoch
        self.list_img_train = glob.glob(os.path.join(dataset_dir, 'train/*/*'))
        self.list_img_test = glob.glob(os.path.join(dataset_dir, 'test/*/*'))
        self.start_idx_train = 0
        self.start_idx_test = 0
        with open('src/config.yaml', 'r') as f:
            param = yaml.load(f, Loader=yaml.FullLoader)
        self.input_size = param['input_size']
        self.code_size = param['code_size']

    def run(self):
        while True:
            try:
                if self.run_flag is False:
                    break
                random.shuffle(self.list_img_train)
                random.shuffle(self.list_img_test)
                time.sleep(self.time_per_epoch)
                # self.model = AE(input_size=self.input_size, code_size=self.code_size)
                self.model = torch.load('runs/ae.pt', map_location=device)
                self.model.eval()
                images = []
                for i in range(self.start_idx_train, self.start_idx_train + 4):
                    image = cv2.imread(self.list_img_train[i])
                    x = self.model.preprocess_image(image)
                    y = self.model(x)
                    x_image = image_torch(x, input_size=self.input_size)
                    y_image = image_torch(y, input_size=self.input_size)
                    if self.cfa_flag:
                        x_image = demosaic.cfa2bgr(x_image)
                        y_image = demosaic.cfa2bgr(y_image)
                    else:
                        x_image = cv2.cvtColor(x_image, cv2.COLOR_GRAY2BGR)
                        y_image = cv2.cvtColor(y_image, cv2.COLOR_GRAY2BGR)
                    one_case = cv2.vconcat([x_image, y_image])
                    images.append(one_case)
                    if len(images) >=4:
                        show1 = concat_image(images, grid_shape=(1, 4), image_size=(self.input_size, self.input_size * 2))

                images = []
                for i in range(self.start_idx_test, self.start_idx_test + 4):
                    image = cv2.imread(self.list_img_test[i])
                    x = self.model.preprocess_image(image)
                    y = self.model(x)
                    x_image = image_torch(x, input_size=self.input_size)
                    y_image = image_torch(y, input_size=self.input_size)
                    if self.cfa_flag:
                        x_image = demosaic.cfa2bgr(x_image)
                        y_image = demosaic.cfa2bgr(y_image)
                    else:
                        x_image = cv2.cvtColor(x_image, cv2.COLOR_GRAY2BGR)
                        y_image = cv2.cvtColor(y_image, cv2.COLOR_GRAY2BGR)

                    one_case = cv2.vconcat([x_image, y_image])
                    images.append(one_case)
                    if len(images) >=4:
                        show2 = concat_image(images, grid_shape=(1, 4), image_size=(self.input_size, self.input_size * 2))
                show = cv2.vconcat([show1, show2])
                self.change_screen_signal.emit(show)
            except Exception:
                print(traceback.format_exc())
                print(sys.exc_info()[2])
                continue


class Home(QWidget, home):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        #initial plot
        self.plot_widget = pg.PlotWidget(self)
        self.plot_widget.setGeometry(20, 20, 371, 371)
        self.plot_widget.plot(y=np.random.normal(size=20))
        self.plot_widget.setTitle("Train Loss")
        self.PlotWindowLabel = QLabel(self)
        self.PlotWindowLabel.setStyleSheet("QLabel{font-size: 40pt; color:rgba(226, 39, 134, 127)}")
        self.loss_hist = []
        self.load_dataset_button.clicked.connect(self.load_dataset)
        self.split_button.clicked.connect(self.split_data)
        self.cfa_convert_button.clicked.connect(self.cfa_data)
        self.gray_convert_button.clicked.connect(self.gray_data)
        self.start_ae_button.clicked.connect(self.train_ae)
        self.stop_ae_button.clicked.connect(self.stop_ae)
        self.start_nn_button.clicked.connect(self.train_nn)
        self.stop_nn_button.clicked.connect(self.stop_nn)
        self.start_svm.clicked.connect(self.train_svm)


    def train_nn(self):
        self.info.append('Traning NN with data at ' + self.dataset_dir.split('/')[-1])
        self.loss_hist = []
        self.plot_widget.clear()
        epochs = int(self.epochs_nn.text())
        batch_size = int(self.batch_nn.text())
        lr = float(self.lr_nn_button.text())
        #create plot 2
        self.train_nn_thread = TrainNeural(self.dataset_dir, epochs, batch_size, lr)
        self.train_nn_thread.change_infor_signal.connect(self.update_info)
        self.train_nn_thread.change_loss_signal.connect(self.update_loss)
        self.train_nn_thread.start()
        self.evalue_nn_thread = EvaluateNeural(self.dataset_dir)
        self.evalue_nn_thread.change_screen_signal.connect(self.update_screen)
        self.evalue_nn_thread.start()


    def train_svm(self):
        self.info.append('Traning SVM with data at ' + self.dataset_dir.split('/')[-1])
        self.plot_widget.clear()
        # create plot 2
        self.train_svm_thread = TrainSVM(self.dataset_dir)
        self.train_svm_thread.change_infor_signal.connect(self.update_info)
        self.train_svm_thread.start()
        self.evalue_svm_thread = EvaluateSVM(self.dataset_dir)
        self.evalue_svm_thread.change_screen_signal.connect(self.update_screen)
        self.evalue_svm_thread.start()

    def stop_nn(self):
        try:
            self.evalue_nn_thread.run_flag = False
            self.train_nn_thread.run_flag = False
            del self.evalue_nn_thread
            del self.train_nn_thread
        except:
            pass

    def load_dataset(self):
        try:
            dir = QFileDialog.getExistingDirectory(None, 'Select', str(ROOT) + '/', QFileDialog.ShowDirsOnly)
            self.dataset_dir = dir
            self.info.append('Load ' + dir.split('/')[-1] + ' completed !!')
        except:
            QMessageBox.warning(self, 'Warning', "Load data failed!")

    def split_data(self):
        from utils.dataset import split_dataset
        try:
            split_dataset(self.dataset_dir, self.dataset_dir + '_split')
            self.dataset_dir = self.dataset_dir + '_split'
            self.info.append('Train test split completed !')
        except:
            pass

    def cfa_data(self):
        from utils.dataset import create_cfa_dataset
        try:
            self.info.append('Converting RGB to CFA ..')
            create_cfa_dataset(self.dataset_dir, self.dataset_dir + '_cfa')
            self.dataset_dir = self.dataset_dir + '_cfa'
            self.info.append('Convert RGB to CFA dataset completed !')
        except:
            pass

    def gray_data(self):
        from utils.dataset import create_gray_dataset
        try:
            self.info.append('Converting RGB to Gray ..')
            create_gray_dataset(self.dataset_dir, self.dataset_dir + '_gray')
            self.dataset_dir = self.dataset_dir + '_gray'
            self.info.append('Convert RGB to gray dataset completed !')
        except:
            pass

    def train_ae(self):
        self.info.append('Traning AE with data at ' + self.dataset_dir.split('/')[-1])
        self.loss_hist = []
        epochs = int(self.epochs_ae.text())
        batch_size = int(self.batch_ae_button.text())
        lr = float(self.lr_ae_button.text())
        self.plot_widget.clear()
        self.train_thread = TrainAutoencoder(self.dataset_dir, epochs, batch_size, lr)
        self.train_thread.change_infor_signal.connect(self.update_info)
        self.train_thread.change_loss_signal.connect(self.update_loss)
        self.train_thread.change_time_signal.connect(self.update_time)
        self.train_thread.start()
        self.evalue_thread = EvaluateAutoencoder(self.dataset_dir, cfa=True)
        self.evalue_thread.change_screen_signal.connect(self.update_screen)
        self.evalue_thread.start()

    def stop_ae(self):
        try:
            self.train_thread.run_flag = False
            self.evalue_thread.run_flag = False

            del self.train_thread
            del self.evalue_thread
        except:
            pass

    def update_info(self, str):
        self.info.append(str)

    def update_loss(self, loss):
        self.loss_hist.append(loss)
        self.plot_widget.plot(self.loss_hist)

    def update_time(self, time):
        self.evalue_thread.time_per_epoch = time

    def update_screen(self, cv_img):
        """Updates the image_label with a new opencv image"""
        qt_img = self.convert_cv_qt(cv_img, 641, 371)
        self.screen.setPixmap(qt_img)

    @staticmethod
    def convert_cv_qt(cv_img, w_screen, h_screen):
        """Convert from an opencv image to QPixmap"""
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        rgb_image = cv2.resize(rgb_image, (w_screen, h_screen))
        #rgb_image = cv2.flip(rgb_image, flipCode=1)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QtGui.QImage(rgb_image.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
        p = convert_to_Qt_format.scaled(w_screen, h_screen, Qt.KeepAspectRatio)
        return QPixmap.fromImage(p)


class Test(QWidget, infference):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        # self.logo_1.setIcon(QIcon('icons/Logo_dhbkdn.jpg'))
        # self.logo_2.setIcon(QIcon('icons/logodtvt.jpg'))
        self.feature_extractor = torch.load('runs/ae.pt', map_location=device)
        self.feature_extractor.eval()
        self.svm_classifier = pickle.load(open('runs/svm.pickle', 'rb'))
        self.nn_classifier = torch.load('runs/classifier.pt', map_location=device)
        self.nn_classifier.eval()
        self.mse = nn.MSELoss()
        self.idx_img = 0
        with open('runs/label.txt', 'r') as f:
            class_name = f.readlines()
            self.class_name = {int(i.split(':')[1].strip('\n')): i.split(':')[0].strip() for i in class_name}
        self.load_button.clicked.connect(self.load_dataset)
        self.next_button.clicked.connect(self.next)
        self.back_button.clicked.connect(self.back)
        self.classifier = 'svm' if self.svm_checker.isChecked() else 'ann'
        with open('src/config.yaml', 'r') as f:
            param = yaml.load(f, Loader=yaml.FullLoader)
        self.input_size = param['input_size']
        self.code_size = param['code_size']

    def load_dataset(self):
        try:
            dir = QFileDialog.getExistingDirectory(None, 'Select', str(ROOT) + '/', QFileDialog.ShowDirsOnly)
            self.dir_image = dir
            self.list_image = glob.glob(dir + '/*.[jp][pn]*')

            image = cv2.imread(self.list_image[self.idx_img])
            image_cfa = demosaic.bgr2cfa(image)
            last_time = time.time()
            x = self.feature_extractor.preprocess_image(image_cfa)
            code = self.feature_extractor.get_coding(x)
            y = self.feature_extractor.decode(code)
            mse = self.mse(x, y).detach().numpy()
            self.re.setText('%.3f'%(mse))
            if self.svm_checker.isChecked():
                index = self.svm_classifier.predict(code.cpu().detach().numpy())[0]
                pred = self.class_name[index]
                self.confidence.setText('')
            else:
                out = self.nn_classifier(code)
                _, index = torch.max(out, 1)
                index = index.tolist()[0]
                percentage = (nn.functional.softmax(out, dim=1)[0] * 100).tolist()
                percentage = percentage[index]
                self.confidence.setText(str(int(percentage)) + '%')
                pred = self.class_name[index]
            inf_time = time.time() - last_time
            self.update_screen(image, self.screen_or)
            re_image = image_torch(y, input_size=self.input_size)
            re_image = demosaic.cfa2bgr(re_image)
            self.update_screen(re_image, self.screen_re)
            self.predict.setText(pred)
            self.time.setText(str(int(inf_time*1000)) + ' ms')
        except:
            QMessageBox.warning(self, 'Warning', "Load data failed!")
    
    def next(self):
        try:
            self.idx_img += 1
            image = cv2.imread(self.list_image[self.idx_img])
            image_cfa = demosaic.bgr2cfa(image)
            last_time = time.time()
            x = self.feature_extractor.preprocess_image(image_cfa)
            code = self.feature_extractor.get_coding(x)
            y = self.feature_extractor.decode(code)
            mse = self.mse(x, y).detach().numpy()
            self.re.setText('%.3f'%(mse))
            if self.svm_checker.isChecked():
                index = self.svm_classifier.predict(code.cpu().detach().numpy())[0]
                pred = self.class_name[index]
                self.confidence.setText('')
            else:
                out = self.nn_classifier(code)
                _, index = torch.max(out, 1)
                index = index.tolist()[0]
                percentage = (nn.functional.softmax(out, dim=1)[0] * 100).tolist()
                percentage = percentage[index]
                self.confidence.setText(str(int(percentage)) + '%')
                pred = self.class_name[index]
            inf_time = time.time() - last_time    
            self.update_screen(image, self.screen_or)
            re_image = image_torch(y, input_size=self.input_size)
            re_image = demosaic.cfa2bgr(re_image)
            self.update_screen(re_image, self.screen_re)
            self.predict.setText(pred)
            self.time.setText(str(int(inf_time*1000)) + ' ms')
        except:
            self.idx_img -= 1
            QMessageBox.warning(self, 'Warning', "Finished!")

    def back(self):
        try:
            self.idx_img -= 1
            image = cv2.imread(self.list_image[self.idx_img])
            image_cfa = demosaic.bgr2cfa(image)
            last_time = time.time()
            x = self.feature_extractor.preprocess_image(image_cfa)
            code = self.feature_extractor.get_coding(x)
            y = self.feature_extractor.decode(code)
            mse = self.mse(x, y).detach().numpy()
            self.re.setText('%.3f'%(mse))
            if self.svm_checker.isChecked():
                index = self.svm_classifier.predict(code.cpu().detach().numpy())[0]
                pred = self.class_name[index]
                self.confidence.setText('')
            else:
                out = self.nn_classifier(code)
                _, index = torch.max(out, 1)
                index = index.tolist()[0]
                percentage = (nn.functional.softmax(out, dim=1)[0] * 100).tolist()
                percentage = percentage[index]
                self.confidence.setText(str(int(percentage)) + '%')
                pred = self.class_name[index]
            inf_time = time.time() - last_time     
            self.update_screen(image, self.screen_or)
            re_image = image_torch(y, input_size=self.input_size)
            re_image = demosaic.cfa2bgr(re_image)
            self.update_screen(re_image, self.screen_re)
            self.predict.setText(pred)
            self.time.setText(str(int(inf_time*1000)) + ' ms')
        except:
            self.idx_img += 1
            QMessageBox.warning(self, 'Warning', "Finished!")

    def update_screen(self, cv_img, screen):
        """Updates the image_label with a new opencv image"""
        qt_img = self.convert_cv_qt(cv_img, 250, 250)
        screen.setPixmap(qt_img)

    @staticmethod
    def convert_cv_qt(cv_img, w_screen, h_screen):
        """Convert from an opencv image to QPixmap"""
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        rgb_image = cv2.resize(rgb_image, (w_screen, h_screen))
        #rgb_image = cv2.flip(rgb_image, flipCode=1)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QtGui.QImage(rgb_image.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
        p = convert_to_Qt_format.scaled(w_screen, h_screen, Qt.KeepAspectRatio)
        return QPixmap.fromImage(p)

        
class MyTableWidget(QWidget):
    def __init__(self):
        super(QWidget, self).__init__()
        self.setGeometry(0, 0, 1920, 1080)
        self.layout = QVBoxLayout(self)
        # Initialize tab screen
        self.tabs = QTabWidget()
        #self.tabs.resize(300, 200)
        self.tab2 = Home()
        self.tab1 = Test()
        # Add tabs
        self.tabs.addTab(self.tab1, "Train")
        self.tabs.addTab(self.tab2, "Infference")
        #############################################
        self.layout.addWidget(self.tabs)
        self.setLayout(self.layout)
        self.setWindowIcon(QtGui.QIcon('src/logomain.jpeg'))

    @pyqtSlot()
    def on_click(self):
        for currentQTableWidgetItem in self.tableWidget.selectedItems():
            print(currentQTableWidgetItem.row(), currentQTableWidgetItem.column(), currentQTableWidgetItem.text())


def main():
    app = QApplication(sys.argv)
    w = MyTableWidget()
    w.show()
    app.exec_()

if __name__ == "__main__":
    main()
