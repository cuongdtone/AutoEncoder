# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'test_tab1.ui'
#
# Created by: PyQt5 UI code generator 5.15.6
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtGui import QIcon


class Ui_Form(object):
    def setupUi(self, Form):
        Form.setObjectName("Form")
        Form.resize(1239, 721)
        Form.setStyleSheet("background-color: rgb(136, 138, 133);")
        self.verticalLayout_3 = QtWidgets.QVBoxLayout(Form)
        self.verticalLayout_3.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_3.setSpacing(0)
        self.verticalLayout_3.setObjectName("verticalLayout_3")
        self.frame_3 = QtWidgets.QFrame(Form)
        self.frame_3.setMaximumSize(QtCore.QSize(16777215, 100))
        self.frame_3.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.frame_3.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_3.setObjectName("frame_3")
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.frame_3)
        self.horizontalLayout.setContentsMargins(9, 9, -1, -1)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.logo_1 = QtWidgets.QLabel(self.frame_3)
        self.logo_1.setMaximumSize(QtCore.QSize(100, 100))
        self.logo_1.setStyleSheet("border-color: rgb(255 ,255,255);\n"
"font: 75 15pt \"Ubuntu Condensed\";\n"
"border-width : 1.5px;\n"
"border-style:inset;\n"
"border-radius: 8px;\n"
"padding: 0 5px;\n"
"color: rgb(255, 255, 255);")
        self.logo_1.setText("")
        self.logo_1.setObjectName("logo_1")
        self.horizontalLayout.addWidget(self.logo_1)
        self.label_15 = QtWidgets.QLabel(self.frame_3)
        self.label_15.setStyleSheet("font: 75 15pt \"Ubuntu Condensed\";\n"
"color: rgb(255, 255, 255)\n"
"")
        self.label_15.setObjectName("label_15")
        self.horizontalLayout.addWidget(self.label_15, 0, QtCore.Qt.AlignHCenter)
        self.logo_2 = QtWidgets.QLabel(self.frame_3)
        self.logo_2.setMaximumSize(QtCore.QSize(100, 100))
        self.logo_2.setStyleSheet("border-color: rgb(255 ,255,255);\n"
"font: 75 15pt \"Ubuntu Condensed\";\n"
"border-width : 1.5px;\n"
"border-style:inset;\n"
"border-radius: 8px;\n"
"padding: 0 5px;\n"
"color: rgb(255, 255, 255);")
        self.logo_2.setText("")
        self.logo_2.setObjectName("logo_2")
        self.horizontalLayout.addWidget(self.logo_2)
        self.verticalLayout_3.addWidget(self.frame_3)
        self.frame_4 = QtWidgets.QFrame(Form)
        self.frame_4.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.frame_4.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_4.setObjectName("frame_4")
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout(self.frame_4)
        self.horizontalLayout_2.setContentsMargins(9, 0, 9, 9)
        self.horizontalLayout_2.setSpacing(9)
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.frame_2 = QtWidgets.QFrame(self.frame_4)
        self.frame_2.setMaximumSize(QtCore.QSize(800, 16777215))
        self.frame_2.setStyleSheet("border-color: rgb(255 ,255,255);\n"
"font: 75 15pt \"Ubuntu Condensed\";\n"
"border-width : 1.5px;\n"
"border-style:inset;\n"
"border-radius: 8px;\n"
"padding: 0 5px;\n"
"color: rgb(255, 255, 255);")
        self.frame_2.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.frame_2.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_2.setObjectName("frame_2")
        self.verticalLayout_4 = QtWidgets.QVBoxLayout(self.frame_2)
        self.verticalLayout_4.setObjectName("verticalLayout_4")
        self.frame_5 = QtWidgets.QFrame(self.frame_2)
        self.frame_5.setStyleSheet("border:none;")
        self.frame_5.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.frame_5.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_5.setObjectName("frame_5")
        self.gridLayout_2 = QtWidgets.QGridLayout(self.frame_5)
        self.gridLayout_2.setContentsMargins(-1, 0, -1, -1)
        self.gridLayout_2.setVerticalSpacing(0)
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.label_5 = QtWidgets.QLabel(self.frame_5)
        self.label_5.setMaximumSize(QtCore.QSize(16777215, 50))
        self.label_5.setObjectName("label_5")
        self.gridLayout_2.addWidget(self.label_5, 0, 0, 1, 1, QtCore.Qt.AlignRight|QtCore.Qt.AlignTop)
        self.label_6 = QtWidgets.QLabel(self.frame_5)
        self.label_6.setMaximumSize(QtCore.QSize(16777215, 50))
        self.label_6.setText("")
        self.label_6.setObjectName("label_6")
        self.gridLayout_2.addWidget(self.label_6, 0, 1, 1, 3)
        self.label_14 = QtWidgets.QLabel(self.frame_5)
        self.label_14.setMaximumSize(QtCore.QSize(16777215, 50))
        self.label_14.setObjectName("label_14")
        self.gridLayout_2.addWidget(self.label_14, 0, 4, 1, 1, QtCore.Qt.AlignLeft|QtCore.Qt.AlignTop)
        self.screen_or = QtWidgets.QLabel(self.frame_5)
        self.screen_or.setMinimumSize(QtCore.QSize(250, 250))
        self.screen_or.setMaximumSize(QtCore.QSize(250, 250))
        self.screen_or.setStyleSheet("background-color: rgb(38, 162, 105);\n"
"border-color: rgb(255 ,255,255);\n"
"font: 75 15pt \"Ubuntu Condensed\";\n"
"border-width : 1.5px;\n"
"border-style:inset;\n"
"border-radius: 8px;\n"
"padding: 0 5px;\n"
"color: rgb(255, 255, 255);")
        self.screen_or.setAlignment(QtCore.Qt.AlignCenter)
        self.screen_or.setObjectName("screen_or")
        self.gridLayout_2.addWidget(self.screen_or, 1, 0, 1, 2, QtCore.Qt.AlignLeft)
        self.line = QtWidgets.QFrame(self.frame_5)
        self.line.setMinimumSize(QtCore.QSize(50, 20))
        self.line.setStyleSheet("border-color: rgb(255 ,255,255);\n"
"font: 75 15pt \"Ubuntu Condensed\";\n"
"background-color: rgb(56, 64, 95);\n"
"border-width : 1.5px;\n"
"border-style:inset;\n"
"border-radius: 8px;\n"
"padding: 0 5px;\n"
"color: rgb(255, 255, 255);")
        self.line.setFrameShadow(QtWidgets.QFrame.Plain)
        self.line.setFrameShape(QtWidgets.QFrame.HLine)
        self.line.setObjectName("line")
        self.gridLayout_2.addWidget(self.line, 1, 2, 1, 1, QtCore.Qt.AlignVCenter)
        self.screen_re = QtWidgets.QLabel(self.frame_5)
        self.screen_re.setMinimumSize(QtCore.QSize(250, 250))
        self.screen_re.setMaximumSize(QtCore.QSize(250, 250))
        self.screen_re.setStyleSheet("background-color: rgb(38, 162, 105);\n"
"border-color: rgb(255 ,255,255);\n"
"font: 75 15pt \"Ubuntu Condensed\";\n"
"border-width : 1.5px;\n"
"border-style:inset;\n"
"border-radius: 8px;\n"
"padding: 0 5px;\n"
"color: rgb(255, 255, 255);")
        self.screen_re.setAlignment(QtCore.Qt.AlignCenter)
        self.screen_re.setObjectName("screen_re")
        self.gridLayout_2.addWidget(self.screen_re, 1, 3, 1, 2, QtCore.Qt.AlignRight)
        self.verticalLayout_4.addWidget(self.frame_5)
        self.frame_6 = QtWidgets.QFrame(self.frame_2)
        self.frame_6.setMaximumSize(QtCore.QSize(16777215, 100))
        self.frame_6.setStyleSheet("border: none;")
        self.frame_6.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.frame_6.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_6.setObjectName("frame_6")
        self.verticalLayout_6 = QtWidgets.QVBoxLayout(self.frame_6)
        self.verticalLayout_6.setObjectName("verticalLayout_6")
        self.frame_11 = QtWidgets.QFrame(self.frame_6)
        self.frame_11.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_11.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_11.setObjectName("frame_11")
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout(self.frame_11)
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        self.back_button = QtWidgets.QPushButton(self.frame_11)
        self.back_button.setMaximumSize(QtCore.QSize(100, 30))
        self.back_button.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.back_button.setStyleSheet("border-color: rgb(255 ,255,255);\n"
"font: 75 15pt \"Ubuntu Condensed\";\n"
"background-color: rgb(56, 64, 95);\n"
"border-width : 1.5px;\n"
"border-style:inset;\n"
"border-radius: 8px;\n"
"padding: 0 5px;\n"
"color: rgb(255, 255, 255);")
        self.back_button.setObjectName("back_button")
        self.horizontalLayout_3.addWidget(self.back_button)
        self.next_button = QtWidgets.QPushButton(self.frame_11)
        self.next_button.setMaximumSize(QtCore.QSize(100, 30))
        self.next_button.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.next_button.setStyleSheet("border-color: rgb(255 ,255,255);\n"
"font: 75 15pt \"Ubuntu Condensed\";\n"
"background-color: rgb(56, 64, 95);\n"
"border-width : 1.5px;\n"
"border-style:inset;\n"
"border-radius: 8px;\n"
"padding: 0 5px;\n"
"color: rgb(255, 255, 255);")
        self.next_button.setObjectName("next_button")
        self.horizontalLayout_3.addWidget(self.next_button)
        self.verticalLayout_6.addWidget(self.frame_11)
        self.load_button = QtWidgets.QToolButton(self.frame_6)
        self.load_button.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.load_button.setStyleSheet("border-color: rgb(255 ,255,255);\n"
"font: 75 15pt \"Ubuntu Condensed\";\n"
"background-color: rgb(56, 64, 95);\n"
"border-width : 1.5px;\n"
"border-style:inset;\n"
"border-radius: 8px;\n"
"padding: 0 5px;\n"
"color: rgb(255, 255, 255);")
        self.load_button.setObjectName("load_button")
        self.verticalLayout_6.addWidget(self.load_button, 0, QtCore.Qt.AlignHCenter)
        self.verticalLayout_4.addWidget(self.frame_6)
        self.label_16 = QtWidgets.QLabel(self.frame_2)
        self.label_16.setMaximumSize(QtCore.QSize(16777215, 20))
        self.label_16.setStyleSheet("font: 75 15pt \"Ubuntu Condensed\";\n"
"color: rgb(255, 255, 255);\n"
"border: none;\n"
"")
        self.label_16.setObjectName("label_16")
        self.verticalLayout_4.addWidget(self.label_16)
        self.horizontalLayout_2.addWidget(self.frame_2)
        self.frame = QtWidgets.QFrame(self.frame_4)
        self.frame.setStyleSheet("border-color: rgb(255 ,255,255);\n"
"font: 75 15pt \"Ubuntu Condensed\";\n"
"border-width : 1.5px;\n"
"border-style:inset;\n"
"border-radius: 8px;\n"
"padding: 0 5px;\n"
"color: rgb(255, 255, 255);")
        self.frame.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.frame.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame.setObjectName("frame")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.frame)
        self.verticalLayout.setObjectName("verticalLayout")
        self.frame_7 = QtWidgets.QFrame(self.frame)
        self.frame_7.setStyleSheet("border: none;")
        self.frame_7.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.frame_7.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_7.setObjectName("frame_7")
        self.verticalLayout_5 = QtWidgets.QVBoxLayout(self.frame_7)
        self.verticalLayout_5.setContentsMargins(0, 0, 0, 50)
        self.verticalLayout_5.setObjectName("verticalLayout_5")
        self.frame_9 = QtWidgets.QFrame(self.frame_7)
        self.frame_9.setMaximumSize(QtCore.QSize(16777215, 70))
        self.frame_9.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.frame_9.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_9.setObjectName("frame_9")
        self.horizontalLayout_5 = QtWidgets.QHBoxLayout(self.frame_9)
        self.horizontalLayout_5.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_5.setObjectName("horizontalLayout_5")
        self.label_9 = QtWidgets.QLabel(self.frame_9)
        self.label_9.setStyleSheet("font: 75 15pt \"Ubuntu Condensed\";\n"
"color: rgb(255, 255, 255);\n"
"border: None")
        self.label_9.setObjectName("label_9")
        self.horizontalLayout_5.addWidget(self.label_9)
        self.svm_checker = QtWidgets.QRadioButton(self.frame_9)
        self.svm_checker.setFocusPolicy(QtCore.Qt.StrongFocus)
        self.svm_checker.setStyleSheet("font: 75 15pt \"Ubuntu Condensed\";\n"
"color: rgb(255, 255, 255);\n"
"border: None")
        self.svm_checker.setChecked(True)
        self.svm_checker.setObjectName("svm_checker")
        self.horizontalLayout_5.addWidget(self.svm_checker)
        self.ann_checker = QtWidgets.QRadioButton(self.frame_9)
        self.ann_checker.setStyleSheet("font: 75 15pt \"Ubuntu Condensed\";\n"
"color: rgb(255, 255, 255);\n"
"border: None")
        self.ann_checker.setObjectName("ann_checker")
        self.horizontalLayout_5.addWidget(self.ann_checker)
        self.verticalLayout_5.addWidget(self.frame_9)
        self.frame_10 = QtWidgets.QFrame(self.frame_7)
        self.frame_10.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.frame_10.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_10.setObjectName("frame_10")
        self.gridLayout = QtWidgets.QGridLayout(self.frame_10)
        self.gridLayout.setHorizontalSpacing(20)
        self.gridLayout.setObjectName("gridLayout")
        self.label_2 = QtWidgets.QLabel(self.frame_10)
        self.label_2.setMinimumSize(QtCore.QSize(0, 50))
        self.label_2.setStyleSheet("font: 75 15pt \"Ubuntu Condensed\";\n"
"color: rgb(255, 255, 255);\n"
"border: None")
        self.label_2.setTextFormat(QtCore.Qt.AutoText)
        self.label_2.setObjectName("label_2")
        self.gridLayout.addWidget(self.label_2, 0, 0, 1, 1, QtCore.Qt.AlignLeft)
        self.predict = QtWidgets.QLabel(self.frame_10)
        self.predict.setMinimumSize(QtCore.QSize(200, 50))
        self.predict.setStyleSheet("border-color: rgb(136, 138, 133);\n"
"background-color: rgb(186, 189, 182);\n"
"border-width : 1.5px;\n"
"border-style:inset;\n"
"border-radius: 8px;\n"
"padding: 0 5px;\n"
"color: rgb(0, 0, 0);")
        self.predict.setText("")
        self.predict.setObjectName("predict")
        self.gridLayout.addWidget(self.predict, 0, 1, 1, 1, QtCore.Qt.AlignLeft)
        self.label = QtWidgets.QLabel(self.frame_10)
        self.label.setMinimumSize(QtCore.QSize(0, 50))
        self.label.setStyleSheet("font: 75 15pt \"Ubuntu Condensed\";\n"
"color: rgb(255, 255, 255);\n"
"border: None")
        self.label.setObjectName("label")
        self.gridLayout.addWidget(self.label, 1, 0, 1, 1, QtCore.Qt.AlignLeft)
        self.confidence = QtWidgets.QLabel(self.frame_10)
        self.confidence.setMinimumSize(QtCore.QSize(200, 50))
        self.confidence.setStyleSheet("border-color: rgb(136, 138, 133);\n"
"background-color: rgb(186, 189, 182);\n"
"border-width : 1.5px;\n"
"border-style:inset;\n"
"border-radius: 8px;\n"
"padding: 0 5px;\n"
"color: rgb(255, 255, 255);")
        self.confidence.setText("")
        self.confidence.setObjectName("confidence")
        self.gridLayout.addWidget(self.confidence, 1, 1, 1, 1, QtCore.Qt.AlignLeft)
        self.label_3 = QtWidgets.QLabel(self.frame_10)
        self.label_3.setMinimumSize(QtCore.QSize(0, 50))
        self.label_3.setStyleSheet("font: 75 15pt \"Ubuntu Condensed\";\n"
"color: rgb(255, 255, 255);\n"
"border: None")
        self.label_3.setObjectName("label_3")
        self.gridLayout.addWidget(self.label_3, 3, 0, 1, 1, QtCore.Qt.AlignLeft)
        self.time = QtWidgets.QLabel(self.frame_10)
        self.time.setMinimumSize(QtCore.QSize(200, 50))
        self.time.setStyleSheet("border-color: rgb(136, 138, 133);\n"
"background-color: rgb(186, 189, 182);\n"
"border-width : 1.5px;\n"
"border-style:inset;\n"
"border-radius: 8px;\n"
"padding: 0 5px;\n"
"color: rgb(255, 255, 255);")
        self.time.setText("")
        self.time.setObjectName("time")
        self.gridLayout.addWidget(self.time, 3, 1, 1, 1, QtCore.Qt.AlignLeft)
        self.label_4 = QtWidgets.QLabel(self.frame_10)
        self.label_4.setMinimumSize(QtCore.QSize(0, 50))
        self.label_4.setStyleSheet("font: 75 15pt \"Ubuntu Condensed\";\n"
"color: rgb(255, 255, 255);\n"
"border: None")
        self.label_4.setObjectName("label_4")
        self.gridLayout.addWidget(self.label_4, 4, 0, 1, 1, QtCore.Qt.AlignLeft)
        self.re = QtWidgets.QLabel(self.frame_10)
        self.re.setMinimumSize(QtCore.QSize(200, 50))
        self.re.setStyleSheet("border-color: rgb(136, 138, 133);\n"
"background-color: rgb(186, 189, 182);\n"
"border-width : 1.5px;\n"
"border-style:inset;\n"
"border-radius: 8px;\n"
"padding: 0 5px;\n"
"color: rgb(255, 255, 255);")
        self.re.setText("")
        self.re.setObjectName("re")
        self.gridLayout.addWidget(self.re, 4, 1, 1, 1, QtCore.Qt.AlignLeft)
        self.verticalLayout_5.addWidget(self.frame_10, 0, QtCore.Qt.AlignTop)
        self.verticalLayout.addWidget(self.frame_7)
        self.frame_8 = QtWidgets.QFrame(self.frame)
        self.frame_8.setMaximumSize(QtCore.QSize(16777215, 150))
        self.frame_8.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.frame_8.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_8.setObjectName("frame_8")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout(self.frame_8)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.label_13 = QtWidgets.QLabel(self.frame_8)
        self.label_13.setStyleSheet("font: 75 8pt \"Ubuntu Condensed\";\n"
"color: rgb(255, 255, 255);\n"
"border: none;")
        self.label_13.setAlignment(QtCore.Qt.AlignCenter)
        self.label_13.setObjectName("label_13")
        self.verticalLayout_2.addWidget(self.label_13)
        self.label_7 = QtWidgets.QLabel(self.frame_8)
        self.label_7.setStyleSheet("font: 75 10pt \"Ubuntu Condensed\";\n"
"border: none;\n"
"color: rgb(255, 255, 255);")
        self.label_7.setObjectName("label_7")
        self.verticalLayout_2.addWidget(self.label_7)
        self.label_8 = QtWidgets.QLabel(self.frame_8)
        self.label_8.setStyleSheet("font: 75 10pt \"Ubuntu Condensed\";\n"
"border: none;\n"
"color: rgb(255, 255, 255);")
        self.label_8.setObjectName("label_8")
        self.verticalLayout_2.addWidget(self.label_8)
        self.label_10 = QtWidgets.QLabel(self.frame_8)
        self.label_10.setStyleSheet("font: 75 10pt \"Ubuntu Condensed\";\n"
"border: none;\n"
"color: rgb(255, 255, 255);")
        self.label_10.setObjectName("label_10")
        self.verticalLayout_2.addWidget(self.label_10)
        self.label_11 = QtWidgets.QLabel(self.frame_8)
        self.label_11.setStyleSheet("font: 75 10pt \"Ubuntu Condensed\";\n"
"border: none;\n"
"color: rgb(255, 255, 255);")
        self.label_11.setObjectName("label_11")
        self.verticalLayout_2.addWidget(self.label_11)
        self.label_12 = QtWidgets.QLabel(self.frame_8)
        self.label_12.setStyleSheet("font: 75 10pt \"Ubuntu Condensed\";\n"
"border: none;\n"
"color: rgb(255, 255, 255);")
        self.label_12.setObjectName("label_12")
        self.verticalLayout_2.addWidget(self.label_12)
        self.verticalLayout.addWidget(self.frame_8, 0, QtCore.Qt.AlignRight)
        self.horizontalLayout_2.addWidget(self.frame)
        self.verticalLayout_3.addWidget(self.frame_4)

        self.retranslateUi(Form)
        QtCore.QMetaObject.connectSlotsByName(Form)

    def retranslateUi(self, Form):
        _translate = QtCore.QCoreApplication.translate
        Form.setWindowTitle(_translate("Form", "Form"))
        self.label_15.setText(_translate("Form", "<html><head/><body><p><span style=\" font-size:24pt; font-weight:600;\">FLOWERS RECOGNITION</span></p></body></html>"))
        self.label_5.setText(_translate("Form", "Original Image"))
        self.label_14.setText(_translate("Form", "Reconstructed Image"))
        self.screen_or.setText(_translate("Form", "Input Image"))
        self.screen_re.setText(_translate("Form", "Recontructed image"))
        self.back_button.setText(_translate("Form", "Back"))
        self.next_button.setText(_translate("Form", "Next"))
        self.load_button.setText(_translate("Form", "Load Image"))
        self.label_16.setText(_translate("Form", "<html><head/><body><p><span style=\" font-size:11pt; font-weight:600;\">22/05/2022</span></p></body></html>"))
        self.label_9.setText(_translate("Form", "<html><head/><body><p><span style=\" font-weight:600;\">Classifier</span></p></body></html>"))
        self.svm_checker.setText(_translate("Form", "SVM"))
        self.ann_checker.setText(_translate("Form", "ANN"))
        self.label_2.setText(_translate("Form", "Prediction"))
        self.label.setText(_translate("Form", "Confidence"))
        self.label_3.setText(_translate("Form", "Inference Time"))
        self.label_4.setText(_translate("Form", "Recontruction Error"))
        self.label_13.setText(_translate("Form", "<html><head/><body><p><span style=\" font-size:14pt; font-weight:600;\">Authors</span></p></body></html>"))
        self.label_7.setText(_translate("Form", "TRAN CHI CUONG"))
        self.label_8.setText(_translate("Form", "DAO DUY NGU"))
        self.label_10.setText(_translate("Form", "LE VAN THIEN"))
        self.label_11.setText(_translate("Form", "NGUYEN VU HOAI DUY"))
        self.label_12.setText(_translate("Form", "HO THANH LONG"))
