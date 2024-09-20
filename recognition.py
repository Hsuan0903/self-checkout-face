import sys  
import os  
import torch    
from library.model.faceDetector import FaceDetector    
from library.model.faceFeatureExtractor import FaceFeatureExtractor    
from library.utils.cameraCapture import CameraCapture  
from library.utils.userRecognition import UserRecognition  
from library.utils.roiDrawer import ROIDrawer  
import cv2    
import numpy as np  

  
if __name__ == "__main__":  
    # 初始化 FaceDetector 和 FaceFeatureExtractor  
    device = 'cuda' if torch.cuda.is_available() else 'cpu'  
    detector = FaceDetector(device=device)  
    feature_extractor = FaceFeatureExtractor(device=device)  
      
    # 初始化 CameraCapture，設定影像大小  
    FRAME_WIDTH = 640  
    FRAME_HEIGHT = 480  
    camera = CameraCapture(camera_index=0, frame_width=FRAME_WIDTH, frame_height=FRAME_HEIGHT)  
      
    # 初始化 ROI Drawer，設定ROI框位置和大小  
    ROI_BOX = (170, 90, 300, 300)  # 格式為 (x, y, w, h)  
    roi_drawer = ROIDrawer(roi_box=ROI_BOX)  
      
    # 初始化 UserRecognition  
    user_recognition = UserRecognition(face_detector=detector, feature_extractor=feature_extractor, roi_drawer=roi_drawer)  
      
    print("Press 'q' to quit.")  
      
    while True:  
        # 從攝像頭捕捉影像  
        frame = camera.get_frame()  
        if frame is None:  
            break  
          
        # 繪製ROI框  
        frame = roi_drawer.draw_roi(frame)  
  
        # 檢測人臉並進行辨識  
        boxes, _ = detector(frame, is_largest_face=False)  
        if boxes is not None:  
            for box in boxes:  
                box = [int(b) for b in box]  
                if user_recognition.is_face_in_roi(box):  
                    print(f"Face in ROI: {box}")  
                    user_name, box = user_recognition.recognize_user(frame)  
                    frame = user_recognition.draw_recognition_result(frame, user_name, box)  
                else:  
                    cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), 2)  
        else:  
            print("No faces detected in frame.")  
  
        # 顯示影像  
        cv2.imshow('Camera', frame)  
          
        key = cv2.waitKey(1) & 0xFF  
        if key == ord('q'):  
            break  
  
    camera.release()  
