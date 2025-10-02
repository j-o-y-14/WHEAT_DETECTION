#  Global Wheat Detection

##  Problem Statement  
Wheat is one of the most important staple crops worldwide, feeding billions of people. However, wheat production is constantly threatened by factors such as pests, diseases, and environmental stress, leading to significant yield losses. Farmers often rely on manual inspection to monitor crop health, which is both time-consuming and prone to human error.  

The objective of this project is to **develop a deep learning model for automated wheat detection and disease classification**. The system should accurately detect wheat heads in large-scale field images and classify potential diseases, providing farmers with timely insights for crop management. By leveraging computer vision techniques and convolutional neural networks (CNNs), the project aims to create a reliable, scalable, and efficient solution that can help:  

- Improve wheat yield forecasting.  
- Enable early detection of crop diseases.  
- Reduce reliance on manual inspection.  
- Support precision agriculture practices through automated monitoring.  

The final goal is not only to **detect wheat** but also to create a deployable tool that empowers farmers with actionable data, ultimately contributing to global food security.  

---

##  Dataset  
The dataset used in this project comes from the **Kaggle Global Wheat Detection Challenge**. It consists of:  

- **train.csv** – contains image IDs, bounding box annotations, and image dimensions.  
- **train.zip** – folder containing training images.  
- **test.zip** – folder containing test images for inference.  
- **sample_submission.csv** – example file in the correct Kaggle submission format.  

In total, the dataset contains **147,793 training images**, making it a large-scale and diverse dataset suitable for deep learning.  

---

##  Project Workflow  

1. **Libraries** – import necessary packages for image preprocessing, model building, training, and visualization.  
2. **Parameters & Paths** – define image size, batch size, number of epochs, and set directories for training and test images.  
3. **Load CSV** – read `train.csv` to link each image with its bounding box annotations.  
4. **Functions** – build helper functions to parse images, normalize them, and convert bounding boxes to YOLO format.  
5. **Split Data** – shuffle and split dataset into **80% training** and **20% validation** for supervised learning.  
6. **Data Augmentation** – apply transformations such as flips, rotations, and zooms (applied only to training images) to improve model generalization.  
7. **Create Datasets** – prepare efficient `tf.data.Dataset` pipelines with batching, shuffling, and prefetching.  
8. **Define Model** – construct a CNN model that predicts bounding box coordinates in YOLO format.  
9. **Train Model** – fit the model on the training data and validate on the validation set. Training was limited to **3 epochs** due to dataset size.  
10. **Plot Loss** – visualize training and validation loss to monitor model performance.  
11. **Predict & Visualize** – run inference on validation and test sets, convert predictions into pixel coordinates, and overlay predicted vs. true bounding boxes on images.  

---

##  Results  
- The model was able to detect wheat heads and draw bounding boxes around them.  
- Validation performance improved after applying augmentation.  
- The final trained model generates bounding box predictions for unseen test images.  

---

##  Deployment  
The final model will be deployed as a **Streamlit web app**, where users can upload wheat field images and get bounding boxes with predictions in real time.  

---

##  Future Improvements  
- Fine-tuning with transfer learning (e.g., EfficientDet, YOLOv5).  
- Training on more epochs with GPUs/TPUs for higher accuracy.  
- Multi-class disease classification alongside wheat detection.  
- Real-world deployment for mobile or drone-based monitoring.  

---

##  Acknowledgements  
- Dataset: [Kaggle Global Wheat Detection](https://www.kaggle.com/c/global-wheat-detection)  
- Libraries: TensorFlow, Keras, OpenCV, Matplotlib  

---
