# Comparison of Two Augmentation Methods in Improving Detection Accuracy of Hemarthrosis

![vgg16](https://github.com/user-attachments/assets/296d25cf-66e8-44af-b6b8-83ab0be83ed2)

**Keywords**: Hemarthrosis, Feature extraction, Cosine similarity, Image recognition, Medical Imaging, Machine learnin

## Introduction
With the increase of computing power, machine learning models in medical imaging have been introduced to help in rending medical diagnosis and inspection, like hemophilia, a rare disorder in which blood cannot clot normally. Often, one of the bottlenecks of detecting hemophilia is the lack of data available to train the algorithm to increase the accuracy. As a possible solution, this research investigated whether introducing augmented data by data synthesis or traditional augmentation techniques can improve model accuracy, helping to diagnose the diseases. 

## Methods
To tackle this research, features of ultrasound images were extracted by the pre-trained VGG-16, and similarities were compared by cosine similarity measure based on extracted features in different distributions among real images, synthetic images, and augmentation images (Real vs. Real, Syn vs. Syn, Real vs. Different Batches of Syn, Real vs. Augmentation Techniques). Model testing performance was investigated using EffientNet-B4 to recognize “blood” images with two augmentation methods. In addition, a gradient-weighted class activation mapping (Grad-CAM) visualization was used to interpret the unexpected results like loss of accuracy. 

## Results
Synthetic and real images do not show high similarity, with a mean similarity score of 0.4737. Synthetic batch 1 dataset and images by horizontal flip are more similar to the original images. Classic augmentation techniques and data synthesis can improve model accuracy, and data by traditional augmentation techniques have a better performance than synthetic data. In addition, the GradCAM heatmap figured out the loss of accuracy is due to a shift in the domain.

## Conclusion
Overall, this research found that two augmentation methods, data synthesis and traditional augmentation techniques, both can improve accuracy to a certain extent to help to diagnose rare diseases.

