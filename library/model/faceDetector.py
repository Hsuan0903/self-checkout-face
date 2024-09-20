import torch  
import torch.nn as nn  
from facenet_pytorch import MTCNN  
from PIL import Image  
import cv2  
import numpy as np  


class FaceDetector(nn.Module):  
    def __init__(self, stride=4, resize=1, margin=14, factor=0.6, keep_all=True, device='cpu'):  
        super(FaceDetector, self).__init__()  
        self.stride = stride  
        self.resize = resize  
        self.device = device  
        self.mtcnn = MTCNN(margin=margin, factor=factor, keep_all=keep_all, device=device)  
  
    def forward(self, image, is_largest_face=False):  
        if isinstance(image, str):  
            image = Image.open(image).convert('RGB')  
        elif isinstance(image, (torch.Tensor, np.ndarray)):  
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  
            image = Image.fromarray(image)  
  
        boxes, probs = self.mtcnn.detect(image)  
  
        faces = []  
        if boxes is not None:  
            if is_largest_face:   
                areas = [(box[2] - box[0]) * (box[3] - box[1]) for box in boxes]  
                max_index = np.argmax(areas)  
                box = boxes[max_index]  
                box = [int(b) for b in box]  
                face = image.crop((box[0], box[1], box[2], box[3]))  
                faces.append(face)  
                boxes = [box]  
            else:  
                for box in boxes:  
                    box = [int(b) for b in box]  
                    face = image.crop((box[0], box[1], box[2], box[3]))  
                    faces.append(face)  
        return boxes, faces  