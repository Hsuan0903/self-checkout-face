import torch  
import torch.nn as nn  
from facenet_pytorch import InceptionResnetV1  
from PIL import Image  
import numpy as np  
import cv2  
import torchvision.transforms as transforms  
  
MODEL = 'vggface2'  # pretrained weight: casia-webface or vggface2

  
class FaceFeatureExtractor(nn.Module):  
    def __init__(self, device='cpu'):  
        super(FaceFeatureExtractor, self).__init__()  
        self.device = device  
        self.model = InceptionResnetV1(pretrained='vggface2').eval().to(device)  
          
        self.transform = transforms.Compose([  
            transforms.Resize((160, 160)),  
            transforms.ToTensor(),  
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  
        ])  
  
    def forward(self, face_image):  
        if isinstance(face_image, str):  
            face_image = Image.open(face_image).convert('RGB')  
        elif isinstance(face_image, (np.ndarray, cv2.Mat)):  
            face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)  
            face_image = Image.fromarray(face_image)  
          
        face_tensor = self.transform(face_image).unsqueeze(0).to(self.device)  
          
        with torch.no_grad():  
            features = self.model(face_tensor)  
          
        return features