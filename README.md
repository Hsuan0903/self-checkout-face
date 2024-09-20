# Face Registration and Recognition System  
   
This is a face registration and recognition system implemented using Python and OpenCV, employing MTCNN for face detection and a feature extractor to extract facial features. The system is divided into two modules: face registration and face recognition.  
   
## Features  
   
- **Face Registration Module**: Captures camera footage, detects faces within a specified Region of Interest (ROI), and stores their feature data.  
- **Face Recognition Module**: Captures camera footage, detects faces within a specified ROI, compares their feature data with registered users, and displays the recognition results.  
   
## Directory Structure  
   
```  
project/  
│  
├── library/  
│   └── model/  
│       ├── faceDetector.py  
│       └── faceFeatureExtractor.py  
│   └── utils/  
│       ├── cameraCapture.py  
│       ├── userRegistration.py  
│       ├── userRecognition.py  
│       └── roiDrawer.py  
├── main.py  
├── main_recognition.py  
├── test.jpg  
```  
   
## Installation and Usage  
   
### 1. Install Dependencies  
   
First, ensure you have Python 3 and pip installed. Then, install the required dependencies:  
   
```sh  
pip install -r requirements.txt  
```  
   
### 2. Setup and Run  
   
#### Face Registration  
   
Run `register.py` to start the face registration module:  
   
```sh  
python register.py  
```  
   
While the program is running, press the `r` key to register a new user and enter the username. Press the `q` key to quit the program.  
   
#### Face Recognition  
   
Run `recognition.py` to start the face recognition module:  
   
```sh  
python recognition.py  
```  
   
The program will automatically detect faces within the ROI and perform recognition. Press the `q` key to quit the program.  
   
## Code Overview  
   
### `faceDetector.py`  
   
Uses MTCNN for face detection, returning face bounding boxes and face images.  
   
### `faceFeatureExtractor.py`  
   
Extracts facial features and generates feature vectors.  
   
### `cameraCapture.py`  
   
Captures camera footage.  
   
### `userRegistration.py`  
   
Handles user registration and stores facial feature data.  
   
### `userRecognition.py`  
   
Handles user recognition, comparing captured facial features with registered users.  
   
### `roiDrawer.py`  
   
Draws the ROI box and checks if a face is within the ROI.  
   
### `register.py`  
   
Main script for face registration.  
   
### `recognition.py`  
   
Main script for face recognition.  
   
## Common Issues  
   
### 1. Faces are not detected?  
   
Ensure the camera is working correctly and there is sufficient lighting. If the issue persists, you may adjust the MTCNN parameters in `faceDetector.py`.  
   
### 2. Recognition results are inaccurate?  
   
Ensure that during the registration process, the face is correctly positioned and clear within the ROI. Also, ensure that during recognition, lighting conditions and face positioning are similar to those during registration.  
   
## Contribution  
   
Feel free to submit issues and pull requests to improve this project.  
   
## License  
   
This project is licensed under the MIT License. See the LICENSE file for details.  
```  
   
This README file includes an introduction to the project, the directory structure, installation and usage instructions, a code overview, common issues, contribution guidelines, and licensing information. I hope this helps you better introduce and use the face registration and recognition system.