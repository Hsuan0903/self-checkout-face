import os  
import cv2  
import torch  
import numpy as np  
from library.model.faceDetector import FaceDetector  
from library.model.faceFeatureExtractor import FaceFeatureExtractor  
from library.utils.roiDrawer import ROIDrawer  
  
class UserRegistration:  
    def __init__(self, face_detector, feature_extractor, roi_drawer, save_path='registered_users'):  
        self.face_detector = face_detector  
        self.feature_extractor = feature_extractor  
        self.roi_drawer = roi_drawer  
        self.save_path = save_path  
        os.makedirs(self.save_path, exist_ok=True)  
  
    def register_user(self, user_name, frame, raw_frame):  
        boxes, faces = self.face_detector(frame, is_largest_face=True) # 只處理最大臉部  
        if faces is not None and len(faces) > 0:  
            box = boxes[0]  
            # 轉換邊界框格式為 (x, y, w, h)  
            converted_box = [box[0], box[1], box[2] - box[0], box[3] - box[1]]  
            if self.roi_drawer.is_face_in_roi(converted_box):  
                face = faces[0]   
                features = self.feature_extractor(face)  
                user_data = {  
                    'user_name': user_name,  
                    'features': features.cpu().numpy()  
                }  
                save_file = os.path.join(self.save_path, f'{user_name}.npy')  
                np.save(save_file, user_data)  
  
                # 儲存註冊影像  
                image_file = os.path.join(self.save_path, f'{user_name}.jpg')  
                cv2.imwrite(image_file, raw_frame)
  
                print(f"User {user_name} registered successfully.")  
                return True  
            else:  
                print("Face not in ROI for registration.")  
        else:  
            print("No face detected for registration.")  
        return False  
      
    def draw_face_bbox(self, frame):  
        boxes, _ = self.face_detector(frame, is_largest_face=False)  
        if boxes is not None:  
            for box in boxes:  
                box = [int(b) for b in box]  
                cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), 2)  
        return frame  
