import cv2  
import time  
  
class CameraCapture:  
    def __init__(self, camera_index=0, frame_width=640, frame_height=480):  
        self.cap = cv2.VideoCapture(camera_index)  
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)  # 設定影像寬度  
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)  # 設定影像高度  
        self.start_time = time.time()  
        self.frame_count = 0  
  
    def get_frame(self):  
        ret, frame = self.cap.read() 
        frame = cv2.flip(frame, 1) 
        if not ret:  
            return None  
  
        self.frame_count += 1  
        elapsed_time = time.time() - self.start_time  
        if elapsed_time > 0:  
            fps = self.frame_count / elapsed_time  
            cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1)  
  
        return frame  
  
    def release(self):  
        self.cap.release()  
        cv2.destroyAllWindows()  
