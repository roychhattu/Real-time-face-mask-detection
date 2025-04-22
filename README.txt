###############################################
March 20, 2025
Real-Time Face Mask Detection using Custom CNN
###############################################
This project is a complete end-to-end pipeline to detect whether a person is wearing a mask or not — from training a CNN model from scratch to real-time inference via webcam, with alerts for non-compliance.

# 1. Project Structure:

|___ FaceMaskDetection.py
├── dataset/
│   ├── with_mask/
│   └── without_mask/
├── haarcascade_frontalface_default.xml
└── README.md


# 2. Requirements:
- pip install numpy opencv-python matplotlib scikit-learn
- winsound is used for beeping alerts — works only on Windows.

# 3. Dataset Format:

dataset/
├── with_mask/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
├── without_mask/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...

- Use .jpg, .jpeg, or .png images.
- Images should be clear frontal faces, ideally at least 60x60 pixels in size.
- Each class folder should ideally have 100–200 images (more = better).


# 4. How to Run: 
- python FaceMaskDetection.py

This will:

- Load and augment the dataset
- Train for 100 epochs using custom CNN
- Plot training & validation loss
- Display a confusion matrix
- Save the trained model as mask_detector_model.pkl
- Automatically launch real-time detection on completion
- If dataset or folders are missing, you’ll get a clear error message.



# 6. Real-Time Detection (Webcam):

Once the model is trained and saved, you can run real-time detection like so:

- Use your webcam feed to detect faces using Haar Cascade
- Predict if the person is wearing a mask
- Draw bounding boxes and show confidence scores
- Play a beep sound if someone is not wearing a mask (Windows only)


# 7. Output Files:


- loss_plot.png: Training vs Validation loss curve.
- confusion_matrix.png: Visual confusion matrix.
- mask_detector_model.pkl: Trained model saved via pickle.


# 8. Model Architecture (Custom CNN):

- Conv2D (16 filters) → ReLU → MaxPool
- Conv2D (32 filters) → ReLU → MaxPool
- Flatten → Fully Connected (128) → ReLU
- Fully Connected (2) → Softmax


 #9. Author: 
            1. Chhattu Roy
            2. Rajni Gandha
            3. Tim Kunyz
            4. Yateen Sakhare
 
