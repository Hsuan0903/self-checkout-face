import cv2  
import torch
import numpy as np  
import os  
import time  
import argparse  
from typing import List, Optional, Tuple  
from facenet_pytorch import MTCNN, InceptionResnetV1


RESCALE_FACTOR = 4
KNOWN_FACE_DIR = "known_faces"
FPS_LIMIT = 15  
DETECTION_INTERVAL = 5
CAMERA_INDEX = 0
IS_SHOW = True
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

class FaceRecognitionApp:  
    def __init__(self, camera_index: int = 0, detection_interval: int = 2, is_show: bool = True):  
        self.video_capture = cv2.VideoCapture(camera_index)
        self.video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)  
        self.video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT) 
        self.known_face_encodings: List[np.ndarray] = []  
        self.known_face_names: List[str] = []  
        self.face_locations: List[Tuple[int, int, int, int]] = []  
        self.face_encodings: List[np.ndarray] = []  
        self.face_names: List[str] = []  
        self.process_this_frame: bool = True  
        self.detection_interval: int = detection_interval  
        self.frame_count: int = 0  
        self.is_show = is_show  
        self.mtcnn = MTCNN(keep_all=True)
        self.resnet = InceptionResnetV1(pretrained='vggface2').eval()


    def load_known_faces(self, face_images_dir: str) -> None:  
        for filename in os.listdir(face_images_dir):  
            if filename.endswith(".jpg") or filename.endswith(".png"):  
                image_path = os.path.join(face_images_dir, filename)  
                image = cv2.imread(image_path)  
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                boxes, _ = self.mtcnn.detect(image)

                if boxes is not None and len(boxes) > 0:
                    largest_box = max(boxes, key=lambda box: (box[2] - box[0]) * (box[3] - box[1]))
                    largest_box = [int(b) for b in largest_box]  

                    face = image[largest_box[1]:largest_box[3], largest_box[0]:largest_box[2]]
                    face_tensor = torch.tensor(face).permute(2, 0, 1).float() / 255.0
                    face_tensor = face_tensor.unsqueeze(0)
                    
                    with torch.no_grad():
                        encodings = self.resnet(face_tensor).detach().cpu()

                    if encodings.size(0) > 0:
                        encoding = encodings[0]  
                        name = os.path.splitext(filename)[0]  
                        self.known_face_encodings.append(encoding.numpy())  
                        self.known_face_names.append(name)  
                else:
                    print(f"No faces found in {filename}. Skipping this image.")

    def register_new_face(self, name: str, image_path: str) -> None:  
        image = cv2.imread(image_path)  
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        boxes, _ = self.mtcnn.detect(image)

        if boxes is not None and len(boxes) > 0:
            largest_box = max(boxes, key=lambda box: (box[2] - box[0]) * (box[3] - box[1]))
            largest_box = [int(b) for b in largest_box]  

            face = image[largest_box[1]:largest_box[3], largest_box[0]:largest_box[2]]
            
            # Resize face to 160x160
            face = cv2.resize(face, (160, 160))
            
            face_tensor = torch.tensor(face).permute(2, 0, 1).float() / 255.0
            face_tensor = face_tensor.unsqueeze(0)

            with torch.no_grad():
                encodings = self.resnet(face_tensor).detach().cpu()
                encoding = encodings[0] 

            self.known_face_encodings.append(encoding.numpy())  
            self.known_face_names.append(name) 

    def get_largest_face_locations(self, frame: np.ndarray) -> Optional[Tuple[int, int, int, int]]:  
        resized_frame = cv2.resize(frame, (0, 0), fx=1/RESCALE_FACTOR, fy=1/RESCALE_FACTOR)  
        boxes, _ = self.mtcnn.detect(resized_frame)
        if not boxes:  
            return None  
        largest_face_location = max(boxes, key=lambda box: (box[2] - box[0]) * (box[3] - box[1]))  
        return largest_face_location 

    def register_face(self) -> None:  
        ret, frame = self.video_capture.read()  
        if ret:  
            frame = cv2.flip(frame, 1)  
            temp_image_path = os.path.join(KNOWN_FACE_DIR, "temp.jpg")  
            cv2.imwrite(temp_image_path, frame)  
            print("Picture taken. Please enter the name for the new face.")  
            name = input("Enter the name for the new face: ")  
            if name:  
                final_image_path = os.path.join(KNOWN_FACE_DIR, f"{name}.jpg")  
                os.rename(temp_image_path, final_image_path)  
                self.register_new_face(name, final_image_path)  
                print(f"New face registered: {name}")  
            else:  
                os.remove(temp_image_path)  
                print("No name entered. Registration cancelled.")

    def process_frame(self, frame: np.ndarray) -> None:  
        resized_frame = cv2.resize(frame, (0, 0), fx=1/RESCALE_FACTOR, fy=1/RESCALE_FACTOR)  
        rgb_resized_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)  
        boxes, _ = self.mtcnn.detect(rgb_resized_frame)  
        self.face_encodings = []
        self.face_names = []
        self.face_locations = []

        if boxes is not None:  
            for box in boxes:
                box = [int(b) for b in box]  # Convert to int
                self.face_locations.append(box)
                face = rgb_resized_frame[box[1]:box[3], box[0]:box[2]]
                
                # Resize face to 160x160
                face = cv2.resize(face, (160, 160))
                
                face_tensor = torch.tensor(face).permute(2, 0, 1).float() / 255.0
                face_tensor = face_tensor.unsqueeze(0)

                # Continue with encoding
                with torch.no_grad():
                    encodings = self.resnet(face_tensor).detach().cpu()

                    if encodings.size(0) > 0:
                        face_encoding = encodings[0]
                        self.face_encodings.append(face_encoding.numpy())
                        name = "Unknown"
                        if self.known_face_encodings:
                            matches = np.linalg.norm(self.known_face_encodings - face_encoding.numpy(), axis=1)
                            best_match_index = np.argmin(matches)
                            if matches[best_match_index] < 0.8:  # Threshold for matching
                                name = self.known_face_names[best_match_index]
                        self.face_names.append(name)

    def display_results(self, frame: np.ndarray) -> None:  
        for (left, top, right, bottom), name in zip(self.face_locations, self.face_names):  
            top *= RESCALE_FACTOR  
            right *= RESCALE_FACTOR  
            bottom *= RESCALE_FACTOR  
            left *= RESCALE_FACTOR  
            color = (0, 0, 255) if name == "Unknown" else (255, 0, 0)
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)  
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), color, cv2.FILLED)  
            cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 1)  

    def run_recognition(self) -> None:  
        fps_counter = 0
        start_time = time.time()
        fps = 0.0  

        while True:  
            ret, frame = self.video_capture.read()  
            frame = cv2.flip(frame, 1)  
            if not ret:  
                break  

            self.frame_count += 1  
            if self.frame_count % self.detection_interval == 0:  
                self.process_frame(frame)  

            self.display_results(frame)  

            fps_counter += 1
            elapsed_time = time.time() - start_time
            if elapsed_time >= 1.0:  
                fps = fps_counter / elapsed_time
                start_time = time.time()
                fps_counter = 0

            cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            if self.is_show:
                cv2.imshow('Video', frame)  

            if cv2.waitKey(int(1000 / FPS_LIMIT)) & 0xFF == ord('q'):  
                break  

        self.video_capture.release()  
        cv2.destroyAllWindows()  

    def run_register(self) -> None:
        while True:
            ret, frame = self.video_capture.read()
            frame = cv2.flip(frame, 1)
            if not ret:
                print("Error: Could not read frame.")
                break
            
            height, width = frame.shape[:2]  
            box_width, box_height = width // 2, height // 2  
            left = (width - box_width) // 2  
            top = (height - box_height) // 2  
            right = left + box_width  
            bottom = top + box_height  
            mask_color = (200, 200, 200)  
            mask = np.zeros_like(frame, dtype=np.uint8)  
            mask[:] = mask_color  
            mask[top:bottom, left:right] = frame[top:bottom, left:right]  
            cv2.rectangle(frame, (left, top), (right, bottom), mask_color, 1)  

            # Display the frame
            cv2.imshow('Register Face', frame)

            # Wait for keypress
            key = cv2.waitKey(1) & 0xFF
            if key == ord('r'):  # Press 'r' to register a new face
                self.register_face()
            elif key == ord('q'):  # Press 'q' to quit
                break

        self.video_capture.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":  
    parser = argparse.ArgumentParser(description="Face Recognition Application")  
    parser.add_argument("--register", action="store_true", help="Run in registration mode")  
    args = parser.parse_args()  

    app = FaceRecognitionApp(CAMERA_INDEX, DETECTION_INTERVAL, IS_SHOW)  
    app.load_known_faces(KNOWN_FACE_DIR)  

    if args.register:  
        app.run_register()  
    else:  
        app.run_recognition()  
