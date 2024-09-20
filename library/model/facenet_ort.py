import onnx
import onnxruntime as ort
import torch

dummy_input = torch.randn(1, 3, 160, 160) 
# 加载 ONNX 模型
onnx_model = onnx.load("facenet.onnx")
onnx.checker.check_model(onnx_model)  # 验证模型

# 使用 ONNX Runtime 进行推理
ort_session = ort.InferenceSession("facenet.onnx")

# 进行推理
def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

onnx_inputs = {ort_session.get_inputs()[0].name: to_numpy(dummy_input)}
onnx_outputs = ort_session.run(None, onnx_inputs)

print(onnx_outputs)
