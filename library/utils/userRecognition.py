import os  
import cv2  
import torch  
import numpy as np  
from library.model.faceDetector import FaceDetector  
from library.model.faceFeatureExtractor import FaceFeatureExtractor  
  
class UserRecognition:  
    def __init__(self, face_detector, feature_extractor, roi_drawer, registered_users_path='registered_users'):  
        self.face_detector = face_detector  
        self.feature_extractor = feature_extractor  
        self.roi_drawer = roi_drawer  
        self.registered_users_path = registered_users_path  
        self.users = self.load_registered_users()  
  
    def load_registered_users(self):  
        users = {}  
        for file_name in os.listdir(self.registered_users_path):  
            if file_name.endswith('.npy'):  
                user_data = np.load(os.path.join(self.registered_users_path, file_name), allow_pickle=True).item()  
                users[user_data['user_name']] = user_data['features']  
        print(f"Loaded users: {users.keys()}")  
        return users  
  
    def recognize_user(self, frame, threshold=0.6):  
        boxes, faces = self.face_detector(frame)  
        if faces is not None and len(faces) > 0:  
            face = faces[0]  # 只處理第一個檢測到的臉部  
            features = self.feature_extractor(face).cpu().numpy()  
            min_distance = float('inf')  
            recognized_user = None  
  
            for user_name, user_features in self.users.items():  
                distance = np.linalg.norm(features - user_features)  
                print(f"Distance to {user_name}: {distance}")  
                if distance < min_distance:  
                    min_distance = distance  
                    recognized_user = user_name  
  
            if min_distance < threshold:  
                print(f"Recognized user: {recognized_user} with distance {min_distance}")  
                return recognized_user, boxes[0]  
            else:  
                print(f"No match found. Minimum distance: {min_distance}")  
                return None, boxes[0]  
         
        return None, None  
  
    def draw_recognition_result(self, frame, user_name, box):  
        if box is not None:  
            left, top, right, bottom = [int(b) for b in box]  
            color = (255, 0, 0) if user_name else (0, 0, 255) 
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)  
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), color, cv2.FILLED)  
            if user_name:  
                cv2.putText(frame, user_name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 1)  
                print(f"Drawing recognition result for: {user_name}")  
            else:  
                cv2.putText(frame, "Unknown", (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 1)  
        return frame  
  
    def is_face_in_roi(self, box):  
        return self.roi_drawer.is_face_in_roi(box)  
