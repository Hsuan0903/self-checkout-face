from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
import numpy as np
import cv2

# Initialize video capture
video_capture = cv2.VideoCapture(0)
video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Initialize MTCNN and InceptionResnetV1
mtcnn = MTCNN(keep_all=True)
resnet = InceptionResnetV1(pretrained='vggface2').eval()

while True:
    ret, frame = video_capture.read()
    if not ret:
        print("Error: Could not read frame.")
        break

    # Convert the frame to RGB for face detection
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Detect faces
    boxes, _ = mtcnn.detect(frame_rgb)

    # Check if any boxes were detected
    if boxes is not None and len(boxes) > 0:
        # Find the largest face by area
        largest_box = max(boxes, key=lambda box: (box[2] - box[0]) * (box[3] - box[1]))
        largest_box = [int(b) for b in largest_box]  # Convert to int

        # Draw rectangle around the largest face
        cv2.rectangle(frame, (largest_box[0], largest_box[1]), (largest_box[2], largest_box[3]), (255, 0, 0), 2)

        # Extract the face region from the original frame
        face = frame_rgb[largest_box[1]:largest_box[3], largest_box[0]:largest_box[2]]

        # Convert face to tensor and normalize
        face_tensor = torch.tensor(face).permute(2, 0, 1).float() / 255.0  # Change to (C, H, W)
        face_tensor = face_tensor.unsqueeze(0)  # Add batch dimension

        # Get embeddings
        with torch.no_grad():
            embeddings = resnet(face_tensor).detach().cpu()
        
        print(embeddings.shape)
        # Display the extracted face
        cv2.imshow('Face', cv2.cvtColor(face, cv2.COLOR_RGB2BGR))  # Convert back to BGR for display

    # Show the video with drawn boxes
    cv2.imshow('Video', frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

# Release resources
video_capture.release()
cv2.destroyAllWindows()
