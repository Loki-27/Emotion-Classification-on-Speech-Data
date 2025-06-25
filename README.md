Emotion Classification on Speech Data
A robust machine learning system for classifying emotions from audio speech using advanced feature extraction and deep neural networks.

Project Overview
  This project implements an end-to-end audio emotion classification system that can identify 7 different emotional states from speech audio files. The system uses a combination of traditional audio features (MFCC, Mel-spectrogram, Chroma) and deep learning techniques to achieve high accuracy in emotion recognition.
  Supported Emotions
  •	Neutral - Calm, emotionless speech
  •	Calm - Peaceful, relaxed tone
  •	Happy - Joyful, positive expressions
  •	Angry - Aggressive, hostile speech
  •	Fearful - Anxious, scared expressions
  •	Disgust - Repulsed, disgusted tone
  •	Surprised - Shocked, unexpected reactions
  Note: The Sad emotion class was intentionally excluded from this model. During initial data analysis, we observed significant overlap between sad and calm emotions in the feature space, leading to classification ambiguity. Removing the sad class improved overall model performance.


Features
  •	Real-time Audio Processing: Process audio files in various formats (WAV, MP3, FLAC, M4A)
  •	Advanced Feature Extraction: Multi-dimensional audio feature extraction including MFCC, spectral features, and chroma
  •	Data Augmentation: Time stretching, pitch shifting, and noise injection for robust training
  •	Deep Learning Architecture: Compact yet powerful neural network with batch normalization and dropout
  •	Web Interface: User-friendly Streamlit application for easy emotion classification
  •	High Accuracy: Achieved competitive performance on emotion recognition benchmarks

Model Training Details
  Dataset Preparation
    •	Emotion Mapping: Original dataset emotions mapped to 7-class system
    •	Class Balance: Maximum 300 files per emotion class to prevent overfitting
    •	Train/Test Split: 80/20 stratified split maintaining class distribution
    •	Augmentation: Applied only to training data (4x increase in training samples)
  Feature Engineering
    The feature extraction process creates a 181-dimensional feature vector combining:
      •	MFCC Features: 39 dimensions (13 coefficients + 13 std + 13 delta means)
      •	Mel-Spectrogram: 48 dimensions (mean values across time)
      •	Spectral Features: 6 dimensions (centroid, rolloff, ZCR statistics)
      •	Chroma Features: 12 dimensions (chromagram means)
  Training Process
      1.	Data Loading: Efficient file-based loading with memory optimization
      2.	Feature Scaling: StandardScaler for zero-mean, unit-variance normalization
      3.	Model Training: Batch training with validation monitoring
      4.	Model Selection: Best model saved based on validation accuracy



Model Performance
  The trained model demonstrates strong performance across all emotion classes:
  •	Training Accuracy: ~95-97%
  •	Test Accuracy: ~80.28%
  Results On Test Data:
  Test Accuracy: 0.8028
  Classification Report:
 Class   	 precision  	  recall 	 f1-score   	support

           0       	0.78    		 0.74  		  0.76      	38
           1      	0.84    		 0.87      	0.85        60
           2    	  0.81   		   0.78      	0.80        60
           3    	  0.81    	   0.85      	0.83        60
           4       	0.77     		 0.78      	0.78        60
           5       	0.87      	 0.69      	0.77        39
           6       	0.75     		 0.87     	0.80        38

    accuracy                          			 0.80       	355
   macro avg       0.80     	 0.80    		  0.80       	 355
weighted avg       0.80   	  0.80     	    0.80       	355
 
Technical Architecture
  Data Preprocessing Pipeline
    1.	Audio Loading:
      o	Sample rate standardization to 16kHz
      o	Duration normalization to 3.5 seconds
      o	Silence trimming (top_db=30)
    2.	Feature Extraction:
      o	MFCC Features: 13 coefficients with mean and standard deviation
      o	MFCC Delta Features: First-order derivatives of MFCC
      o	Mel-Spectrogram: 48 mel bands converted to dB scale
      o	Spectral Features: Centroid, rolloff, and zero-crossing rate
      o	Chroma Features: 12-dimensional chromagram
    3.	Data Augmentation:
      o	Time stretching (0.9x rate)
      o	Pitch shifting (+2 semitones)
      o	Gaussian noise addition (0.5% of signal amplitude)


Model Architecture
  Dense Neural Network:
  ├── Input Layer (Feature dimension)
  ├── Dense Layer (512 neurons) + ReLU + BatchNorm + Dropout(0.3)
  ├── Dense Layer (256 neurons) + ReLU + BatchNorm + Dropout(0.3)  
  ├── Dense Layer (128 neurons) + ReLU + BatchNorm + Dropout(0.2)
  ├── Dense Layer (64 neurons) + ReLU + Dropout(0.2)
  └── Output Layer (7 classes) + Softmax

  
Key Design Decisions:
  •	Batch Normalization: Stabilizes training and improves convergence
  •	Progressive Dropout: Higher dropout in early layers, lower in later layers
  •	Compact Architecture: Efficient for deployment while maintaining performance
Training Configuration
  •	Optimizer: Adam (learning_rate=0.001)
  •	Loss Function: Sparse Categorical Crossentropy
  •	Batch Size: 64
  •	Max Epochs: 100
  •	Callbacks: 
    o	ModelCheckpoint (save best model)
    o	EarlyStopping (patience=10)
    o	ReduceLROnPlateau (factor=0.5, patience=5)

Deployment
  Streamlit Cloud Deployment
    1.	Prepare repository:
      o	Ensure all files are committed to Git
      o	Include requirements.txt with all dependencies
  2.	Deploy to Streamlit Cloud:
      o	Connectd my GitHub repository to Streamlit Cloud
      o	Configure the main file path as app.py
      o	Then Deployed


Access Website at:
		https://emotion-classifier-vhm.streamlit.app/
