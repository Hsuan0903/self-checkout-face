import numpy as np
import cv2

def pnet_preprocess(image, scale_factor=0.6):
    """
    P-Net前处理，缩放输入图像并进行归一化。
    """
    # 缩放图像
    h, w, _ = image.shape
    new_h, new_w = int(h * scale_factor), int(w * scale_factor)
    resized_image = cv2.resize(image, (new_w, new_h))

    # 转换为适合P-Net的输入：维度 (1, 3, h, w)，并进行归一化处理
    resized_image = np.transpose(resized_image, (2, 0, 1))  # 转换为 (3, h, w)
    resized_image = resized_image[np.newaxis, :]  # 扩展为 (1, 3, h, w)
    resized_image = resized_image.astype(np.float32) / 255.0  # 归一化到 [0, 1]

    return resized_image

def pnet_postprocess(prob, bbox, threshold=0.6):
    """
    P-Net 后处理，基于置信度的候选框筛选和NMS。
    """
    # 置信度筛选
    indices = np.where(prob[0, 1, :, :] > threshold)
    
    # 将候选框转换为坐标，并使用NMS过滤
    boxes = bbox_to_boxes(indices, bbox)  # 自定义函数将indices转换为坐标
    selected_boxes = nms(boxes, prob)  # 执行非极大值抑制

    return selected_boxes


def rnet_preprocess(image, boxes):
    """
    R-Net前处理，将P-Net检测到的候选框区域裁剪并调整为24x24大小。
    """
    cropped_images = []
    for box in boxes:
        # 从图像中裁剪出候选框区域
        cropped_image = image[box[1]:box[3], box[0]:box[2]]
        resized_image = cv2.resize(cropped_image, (24, 24))  # R-Net输入大小24x24
        cropped_images.append(np.transpose(resized_image, (2, 0, 1)))  # (3, h, w)

    # 扩展为batch size的维度 (batch_size, 3, 24, 24)
    cropped_images = np.stack(cropped_images).astype(np.float32) / 255.0  # 归一化

    return cropped_images


def rnet_postprocess(prob, bbox, threshold=0.7):
    """
    R-Net后处理，基于R-Net置信度输出进一步筛选候选框。
    """
    # 筛选出置信度高于阈值的候选框
    indices = np.where(prob[:, 1] > threshold)
    
    selected_boxes = bbox[indices]  # 提取相应的候选框

    return selected_boxes


def onet_preprocess(image, boxes):
    """
    O-Net前处理，将R-Net输出的候选框区域裁剪并调整为48x48大小。
    """
    cropped_images = []
    for box in boxes:
        # 从图像中裁剪出候选框区域
        cropped_image = image[box[1]:box[3], box[0]:box[2]]
        resized_image = cv2.resize(cropped_image, (48, 48))  # O-Net输入大小48x48
        cropped_images.append(np.transpose(resized_image, (2, 0, 1)))  # (3, h, w)

    # 扩展为batch size的维度 (batch_size, 3, 48, 48)
    cropped_images = np.stack(cropped_images).astype(np.float32) / 255.0  # 归一化

    return cropped_images


def onet_postprocess(prob, bbox, landmarks, threshold=0.8):
    """
    O-Net后处理，基于O-Net输出的置信度和关键点坐标处理最终的候选框。
    """
    # 筛选出置信度高于阈值的候选框
    indices = np.where(prob[:, 1] > threshold)

    selected_boxes = bbox[indices]  # 提取相应的候选框
    selected_landmarks = landmarks[indices]  # 提取对应的关键点

    return selected_boxes, selected_landmarks


def run_mtcnn(image, pnet_session, rnet_session, onet_session):
    # 1. P-Net 前处理
    pnet_input = pnet_preprocess(image)
    
    # 2. P-Net 推理
    pnet_outputs = pnet_session.run(None, {pnet_session.get_inputs()[0].name: pnet_input})
    pnet_prob, pnet_bbox = pnet_outputs
    
    # 3. P-Net 后处理
    pnet_boxes = pnet_postprocess(pnet_prob, pnet_bbox)
    
    # 4. R-Net 前处理
    rnet_input = rnet_preprocess(image, pnet_boxes)
    
    # 5. R-Net 推理
    rnet_outputs = rnet_session.run(None, {rnet_session.get_inputs()[0].name: rnet_input})
    rnet_prob, rnet_bbox = rnet_outputs
    
    # 6. R-Net 后处理
    rnet_boxes = rnet_postprocess(rnet_prob, rnet_bbox)
    
    # 7. O-Net 前处理
    onet_input = onet_preprocess(image, rnet_boxes)
    
    # 8. O-Net 推理
    onet_outputs = onet_session.run(None, {onet_session.get_inputs()[0].name: onet_input})
    onet_prob, onet_bbox, onet_landmarks = onet_outputs
    
    # 9. O-Net 后处理
    final_boxes, final_landmarks = onet_postprocess(onet_prob, onet_bbox, onet_landmarks)
    
    return final_boxes, final_landmarks


image = cv2.imread('path_to_image.jpg')
final_boxes, final_landmarks = run_mtcnn(image, pnet_session, rnet_session, onet_session)

print("Final detected boxes:", final_boxes)
print("Detected landmarks:", final_landmarks)
