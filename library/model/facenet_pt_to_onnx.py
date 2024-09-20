import torch
from facenet_pytorch import InceptionResnetV1


model = InceptionResnetV1(pretrained='vggface2').eval()

dummy_input = torch.randn(1, 3, 160, 160)  

torch.onnx.export(model, 
                  dummy_input, 
                  "weight/facenet.onnx", 
                  opset_version=12)
