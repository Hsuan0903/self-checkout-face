import torch
from facenet_pytorch import MTCNN


mtcnn = MTCNN(keep_all=True, device='cuda' if torch.cuda.is_available() else 'cpu')


pnet_dummy_input = torch.randn(1, 3, 12, 12) # P-Net

# P-Net export onnx format
torch.onnx.export(mtcnn.pnet, 
                  pnet_dummy_input, 
                  "weight/pnet.onnx", 
                  input_names=['input'], 
                  output_names=['prob', 'bbox'],  # P-Net output is prob and bbox
                  dynamic_axes={'input': {0: 'batch_size'}, 'prob': {0: 'batch_size'}, 'bbox': {0: 'batch_size'}},
                  opset_version=12)

rnet_dummy_input = torch.randn(1, 3, 24, 24)  # R-Net

# R-Net export onnx format
torch.onnx.export(mtcnn.rnet, 
                  rnet_dummy_input, 
                  "weight/rnet.onnx", 
                  input_names=['input'], 
                  output_names=['prob', 'bbox'],  # P-Net output is prob and bbox
                  dynamic_axes={'input': {0: 'batch_size'}, 'prob': {0: 'batch_size'}, 'bbox': {0: 'batch_size'}},
                  opset_version=12)


onet_dummy_input = torch.randn(1, 3, 48, 48)  # O-Net

# O-Net export onnx format
torch.onnx.export(mtcnn.onet, 
                  onet_dummy_input, 
                  "weight/onet.onnx", 
                  input_names=['input'], 
                  output_names=['prob', 'bbox', 'landmarks'],  # O-Net output is prob and bboxs and landmarks
                  dynamic_axes={'input': {0: 'batch_size'}, 'prob': {0: 'batch_size'}, 'bbox': {0: 'batch_size'}, 'landmarks': {0: 'batch_size'}},
                  opset_version=12)


# 保存 P-Net、R-Net 和 O-Net 的權重  
torch.save(mtcnn.pnet.state_dict(), 'weight/pnet.pt')  
torch.save(mtcnn.rnet.state_dict(), 'weight/rnet.pt')
torch.save(mtcnn.onet.state_dict(), 'weight/onet.pt')