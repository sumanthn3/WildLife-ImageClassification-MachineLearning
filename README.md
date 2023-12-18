
# Machine Learning

# Classifying Wildlife Species from

# Camera Trap Images


**Team:**

Sumanth Nandeti (G40560437)

Harshini Mandalapu

Darshil Shah

**Problem Definition**

The manual analysis of massive sets of camera trap images for wildlife monitoring is time-consuming and often inaccurate, leading to delays in vital conservation efforts. This inefficiency hinders the timely insights necessary for protecting endangered species and maintaining biodiversity.

**Objective**

The goal is to develop a model that can help detect if a specific species is in the image or not. We are planning to use a model based on Convolutional neural network architecture which is suitable for object detection and image classification.This solution drastically reduces the time and resources needed for data analysis, providing quick, actionable insights for effective conservation strategies.

**Dataset Description**

The data set provides labelled training data for wildlife species identification. The dataset has 20,000 images and each image is paired with a unique identifier, labels and the site from where the image was taken. Data has label classes of 8 species

**Data Exploration**

- There are 20,000 images in the dataset. 
- Most of the image dimensions were of size 640 X 340 pixels. 
- As per the labels given, in the images Antelope-Duiker and Rodent appear to have the highest counts, nearing 2500 images. Bird, Blank, Leopard, and Monkey-Posimian show fairly consistent counts, ranging between 1500 and 2000 images. Civet-Genet stands out with the lowest count, well below 1000. Hog is moderately represented, with its count just above 1000. Images are taken over 148 different sites. 
- Training and Test set images were taken of different sites, and there is no overlap between them.


|![](Aspose.Words.24c3abc6-11c7-48c3-935b-0321a81e694e.001.jpeg)**Fig.1 Classification of Animals**|![](Aspose.Words.24c3abc6-11c7-48c3-935b-0321a81e694e.002.jpeg)**Fig.2 Site Source of Images**|
| :-: | :-: |


**Data Visualization**


|![](Aspose.Words.24c3abc6-11c7-48c3-935b-0321a81e694e.003.jpeg)**Fig.3 Displaying Images of Leopard**|![](Aspose.Words.24c3abc6-11c7-48c3-935b-0321a81e694e.004.jpeg)**Fig.4 Displaying Images of Monkey\_Prosimian**|
| :-: | :-: |

**Experimental Setup**:

Model Architectures: We experimented with Six different models:

1. ResNet50 - **Baseline**
1. ResNet152
1. Ensemble (EfficientB0 + ResNet50)
1. EfficientNetB4
1. EfficientNetV2\_S
1. Ensemble (EfficientNetB4 L1L2 + EfficientNetV2\_S) - **Final Model**

Architecture of ResNet50 - Base Model:

![](Aspose.Words.24c3abc6-11c7-48c3-935b-0321a81e694e.005.png)

**Fig.5 Architecture of ResNet50**






Architecture EfficientNetV2\_S:

![](Aspose.Words.24c3abc6-11c7-48c3-935b-0321a81e694e.006.png)

Architecture EfficientNetB4:

![](Aspose.Words.24c3abc6-11c7-48c3-935b-0321a81e694e.007.png)

**Fig.6 Architecture of EfficientNet B4**


|**Feature/Model**|**ResNet50**|**ResNet152**|**EfficientNetB0**|**EfficientNetB4**|**EfficientNetV2S**|
| :- | :- | :- | :- | :- | :- |
|Total Layers|50|152|Variable (Compound Scaling)|Variable (Compound Scaling)|Variable (EfficientNet Design Advances)|
|Architectural Feature|Residual Connections|Residual Connections|Compound Scaling Method|Compound Scaling Method|Improved Fused-MBConv Blocks|
|Depth Layers|Deep|Deeper|Moderate|Deeper than B0|<p>Optimized for</p><p>Speed</p>|
|Base Model Usage|Pre-trained on ImageNet,|Pre-trained on ImageNet,|Pre-trained on ImageNet|Pre-trained on ImageNet|Pre-trained on ImageNet|
|Training Image Size|224x224|224x224|224x224|384x384|366x366|
|Output Layer|Dense softmax layer (class number 8)|Dense softmax layer (class number 8)|Dense layer with 8 units, softmax|Dense layer with 8 units, softmax|Dense layer with 8 units, softmax|
|Data Augmentation|Not specified|Not specified|Rotation, shift, shear, zoom, flip, brightness adjustment|Rotation, shift, shear, zoom, flip, brightness adjustment|Rotation, shift, shear, zoom, flip, brightness adjustment|
|Fine-Tuning|Last 20 layers unfrozen|Last 20 layers unfrozen|-|-|-|

**Model Training:**

The models were trained for 10 epochs with a batch size of 32. We used Adam optimizer and a sparse categorical cross-entropy loss function to optimize the model.




|ResNet50 - **Baseline**|Ensemble (EfficientNetB4 L1L2 + EfficientNetV2\_S) - **Final Model**|
| :-: | :-: |
|![](Aspose.Words.24c3abc6-11c7-48c3-935b-0321a81e694e.008.png)**Fig.7 Training and validation loss**|![](Aspose.Words.24c3abc6-11c7-48c3-935b-0321a81e694e.009.png)**Fig.8 Training and validation loss**|
|![](Aspose.Words.24c3abc6-11c7-48c3-935b-0321a81e694e.010.png)**Fig.9 Log Loss**|<p>![](Aspose.Words.24c3abc6-11c7-48c3-935b-0321a81e694e.011.png)</p><p>**Fig.10 Log Loss**</p>|

After the base EfficientNetB4 model, a GlobalAveragePooling2D layer is applied. This layer reduces the dimensionality of the output from the convolutional layers, condensing the feature information into a single vector per image. A Dense layer with eight neurons (presumably for eight classes) with softmax activation is used for classification. The softmax function is standard for multi-class classification tasks, outputting a probability distribution over the classes.

The Final model leverages the efficient and scalable architecture of EfficientNetB4, tailored for the specific task through fine-tuning. Data augmentation techniques enhance the model's ability to generalize, while early stopping and model checkpointing make the training process more efficient and effective. The end goal is to create a model capable of accurately classifying images into one of 8 categories, benefiting from the advanced features of EfficientNetB4.

The model EfficientNetV2S architecture, is known for its balance of efficiency and accuracy in image classification tasks. Using data augmentation techniques aims to improve the model's ability to generalize to new data. The training process is designed to effectively leverage the capabilities of EfficientNetV2S, adapting it to the specific characteristics of the dataset at hand.



The two graphs compare the training and validation loss curves of the two models during the training process. The baseline model shows the training loss initially higher than the validation loss until the 2nd epoch. After the 2nd epoch, the training loss continues to decrease.

**Evaluation Metric**:

We evaluated models based on accuracy and Log Loss. EfficientNetB4 with L1 and L2 regularization showed minimal Log loss and promising accuracy compared to the baseline model and the Ensemble models we experimented with. 

|**Model/Feature**|**Training Accuracy**|**Validation Accuracy**|**Validation Loss**|**Log Loss**|
| :- | :- | :- | :- | :- |
|**ResNet50**|81%|79%|0\.63|2\.02|
|**ResNet152**|82%|79%|0\.6|1\.93|
|**Ensemble Model (EfficientNetB0 + ResNet50)**|98%|89%|0\.4|2\.44|
|<p>**Ensemble Model (B0+ R50)**</p><p>**L1L2 Regularized**</p>|94%|89%|0\.402|2\.14|
|**EfficientNetB4**|85%|84%|0\.44|1\.84|
|**EfficientNetV2S**|83%|88%|0\.34|1\.836|
|**EfficientNetB4 L1L2**|84%|85%|0\.43|1\.704|

We have observed the F1-Score of the three main models which we used in this project. We have achieved promising results and their F1-scores are shown below respectively.

|**Model**|**F1-Score**|
| :- | :- |
|EfficientNetB4|0\.89|
|EfficientNetV2S|0\.88|
|EfficientNetB4 L1L2|0\.90|

**Confusion Matrix**

Displaying the confusion matrix of validation data. since we don't have true labels of test data. We used the validation data to form the confusion matrix

![](Aspose.Words.24c3abc6-11c7-48c3-935b-0321a81e694e.012.png)

**Fig.11 Confusion Matrix**

**Conclusion and Future work**

In conclusion, the EfficientNetB4 with L1 and L2 regularization displayed better performance by achieving optimum Log-loss in comparison to other models. For future work, we can leverage transformers to perform species classification from images. We may use vision transformers or Swin transformers to achieve effective species detection from images.
