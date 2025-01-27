# Multimodal-Fusion-for-Person-Identification

## Overview
This project explores biometric person identification using multimodal fusion techniques, combining **facial features** and **fingerprints** to enhance identification accuracy. By leveraging deep learning (VGG16 architecture), the system achieves robust and reliable results for security-sensitive applications.
## Motivation
This is my first project in machine learning. I conducted a literature survey to understand various models and approaches for biometric systems. I chose to implement the **VGG16 architecture** for feature extraction because of its simplicity and effectiveness. This project helped me learn the fundamentals of machine learning, data processing, and model evaluation.

## Methodology
1. **Datasets**:
   - **SOCOFing Dataset**: Contains 6,000 fingerprint images.
   - **Synthetic Face Dataset**: 10,000 high-resolution synthetic face images.

2. **Model**:
   - **VGG16 Architecture**: Used for feature extraction from face and fingerprint images.
   - Features from both modalities were fused to create a combined representation.
## VGG16 Architecture
![VGG16 Architecture](images/vgg16_architecture.png "VGG16 Diagram")

3. **Tools and Technologies**:
   - Python, TensorFlow, Keras, NumPy, Matplotlib.
  ## Results
- Achieved **99.99% accuracy** in identifying individuals using fused face and fingerprint features.
- Successfully classified authorized and non-authorized users.
- Generated a confusion matrix and detailed evaluation metrics (precision, recall, F1-score).

### Sample Results:
- **Authorized User Prediction**: Correctly identified individuals with matching face and fingerprint features.
- **Non-Authorized User Prediction**: Successfully flagged individuals without valid biometrics.
## Learnings
- How to preprocess and manage multimodal data for machine learning tasks.
- The importance of feature extraction and fusion in improving system performance.
- Basics of deep learning architectures like VGG16.
- Evaluating machine learning models using metrics like accuracy and confusion matrices.
## References
1. SOCOFing Dataset for fingerprint images.
2. "Very Deep Convolutional Networks for Large-Scale Image Recognition" by Simonyan and Zisserman (VGG16 paper).
3. Papers and literature survey documents included in `REPORT.pdf`.

## License
This project is licensed under the MIT License. See the LICENSE file for details.



