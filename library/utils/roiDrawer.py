import cv2  
  
class ROIDrawer:  
    def __init__(self, roi_box):  
        self.roi_box = roi_box  # 格式為 (x, y, w, h)  
  
    def draw_roi(self, frame):  
        x, y, w, h = self.roi_box  
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)  
        return frame  
  
    def is_face_in_roi(self, box):  
        x, y, w, h = self.roi_box  
        bx, by, bx2, by2 = box  
        bw = bx2 - bx
        bh = by2 - by
  
        # 檢查邊界框是否完全在ROI框內  
        return (bx >= x and by >= y and bx + bw <= x + w and by + bh <= y + h)  
