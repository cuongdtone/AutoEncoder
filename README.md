# Flower Recognition with Auto Encoder 
***
Final Project of Machine Learning course  
Lecture: Tran Thi Minh Hanh
## Member
```buildoutcfg
Tran Chi Cuong
Le Van Thien
Dao Duy Ngu
Nguyen Vu Hoai Duy
Ho Thanh Long
```
## Dataset  

[Link Dataset](https://drive.google.com/drive/folders/1H78orfSRfSBBxQyaxMgj_ZCKmp61eCDC?usp=sharing)
## Report
[Report](https://drive.google.com/file/d/1uIyUPaEA5EE7FDbGGl6aoHODTnk0uyWK/view?usp=sharing)  
[Slice](https://drive.google.com/file/d/1uxdaAadKE7QhhAX5leOSYxQUVOMccNx3/view?usp=sharing)


## Auto Encoder 
- Left: Original Image, Right: Recontructed Image (Test Set)  
   
<img  src="imgs/test_org_1.jpg" alt="Trulli" style="width:300px;">--><img  src="imgs/test_pred_1.jpg" alt="Trulli" style="width:300px;">   
  
-  

<img  src="imgs/test_org_2.jpg" alt="Trulli" style="width:300px;"> --><img  src="imgs/test_pred_2.jpg" alt="Trulli" style="width:300px;">    
 
-  

<img  src="imgs/test_org_3.jpg" alt="Trulli" style="width:300px;"> --><img  src="imgs/test_pred_3.jpg" alt="Trulli" style="width:300px;">  

## Classification  

<figure>
<img  src="runs/confusion_matrix_nn_test.png" alt="Trulli" style="width:720px;">
<figcaption align = "center"><b>Confusion Matrix in Test Set (ANN)</b></figcaption>
</figure>

<figure>
<img src="runs/confusion_matrix_svm_test.png" alt="Trulli" style="width:720px;">
<figcaption align = "center"><b>Confusion Matrix in Test Set (SVM)</b></figcaption>
</figure>

<figure>
<img src="runs/confusion_matrix_nn_train.png" alt="Trulli" style="width:720px;">
<figcaption align = "center"><b>Confusion Matrix in Train Set (ANN)</b></figcaption>
</figure>

<figure>
<img src="runs/confusion_matrix_svm_train.png" alt="Trulli" style="width:720px;">
<figcaption align = "center"><b>Confusion Matrix in Train Set (SVM)</b></figcaption>
</figure>
