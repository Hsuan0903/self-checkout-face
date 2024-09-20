import sys  
import os  
import torch    
from library.model.faceDetector import FaceDetector    
from library.model.faceFeatureExtractor import FaceFeatureExtractor    
from library.utils.cameraCapture import CameraCapture  
from library.utils.userRegistration import UserRegistration  
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
    ROI_BOX = (170, 90, 300, 300)   # 格式為 (x, y, w, h)  
    roi_drawer = ROIDrawer(roi_box=ROI_BOX)  
      
    # 初始化 UserRegistration  
    user_registration = UserRegistration(face_detector=detector, feature_extractor=feature_extractor, roi_drawer=roi_drawer)  
      
    print("Press 'r' to register a new user, 'q' to quit.")  
      
    while True:  
        # 從攝像頭捕捉影像  
        frame = camera.get_frame()  
        raw_frame = frame.copy()
        if frame is None:  
            break  
          
        # 繪製ROI框  
        frame = roi_drawer.draw_roi(frame)  
  
        # 繪製人臉BBOX  
        frame = user_registration.draw_face_bbox(frame)  
          
        # 顯示影像  
        cv2.imshow('Camera', frame)  
          
        key = cv2.waitKey(1) & 0xFF  
        if key == ord('r'):  
            user_name = input("Enter user name: ")  
            if user_registration.register_user(user_name, frame, raw_frame):  
                # 註冊成功時的視覺提示  
                for _ in range(3):  
                    cv2.putText(frame, "Registered!", (200, 200), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)   
                    frame = roi_drawer.draw_roi(frame)  
                    frame = user_registration.draw_face_bbox(frame)  
                    cv2.imshow('Camera', frame)  
                    cv2.waitKey(500)  
        elif key == ord('q'):  
            break  
  
    camera.release()  
