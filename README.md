# Flower Recognition with Auto Encoder 
***
Final Project of Machine Learning course
## Member
```buildoutcfg
Tran Chi Cuong
Le Van Thien
Dao Duy Ngu
Nguyen Vu Hoai Duy
Ho Thanh Long
```
## Sumary  
***
- Flower recognition with Auto Encoder + ANN/SVM

## Auto Encoder 
- Left: Original Image, Right: Recontructed Image (Test Set)  
   
![](imgs/test_org_1.jpg)     --->  ![](imgs/test_pred_1.jpg)  
![](imgs/test_org_2.jpg)     --->  ![](imgs/test_pred_2.jpg)  
![](imgs/test_org_3.jpg)     --->  ![](imgs/test_pred_3.jpg)  

## Classification  

<figure>
<img  src="runs/confusion_matrix_nn_test.png" alt="Trulli" style="width:1080px;">
<figcaption align = "center"><b>Confusion Matrix in Test Set (ANN)</b></figcaption>
</figure>

<figure>
<img src="runs/confusion_matrix_svm_test.png" alt="Trulli" style="width:1080px;">
<figcaption align = "center"><b>Confusion Matrix in Test Set (SVM)</b></figcaption>
</figure>

<figure>
<img src="runs/confusion_matrix_nn_train.png" alt="Trulli" style="width:1080px;">
<figcaption align = "center"><b>Confusion Matrix in Train Set (ANN)</b></figcaption>
</figure>

<figure>
<img src="runs/confusion_matrix_svm_train.png" alt="Trulli" style="width:1080px;">
<figcaption align = "center"><b>Confusion Matrix in Train Set (SVM)</b></figcaption>
</figure>