import cv2  
import face_recognition  
import numpy as np  
import os  
import time  
import argparse  # Import argparse for command-line parsing
from typing import List, Optional, Tuple  

RESCALE_FACTOR = 4
KNOWN_FACE_DIR = "known_faces"
FPS_LIMIT = 15  # Set desired FPS limit
DETECTION_INTERVAL = 5
CAMERA_INDEX = 0
IS_SHOW = True
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

class FaceRecognitionApp:  
    def __init__(self, camera_index: int = 0, detection_interval: int = 2, is_show: bool = True):  
        self.video_capture = cv2.VideoCapture(camera_index)
        self.video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)  # Lower resolution
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

    def load_known_faces(self, face_images_dir: str) -> None:  
        for filename in os.listdir(face_images_dir):  
            if filename.endswith(".jpg") or filename.endswith(".png"):  
                image_path = os.path.join(face_images_dir, filename)  
                image = face_recognition.load_image_file(image_path)  
                encodings = face_recognition.face_encodings(image)
                if encodings:  
                    encoding = encodings[0]  
                    name = os.path.splitext(filename)[0]  
                    self.known_face_encodings.append(encoding)  
                    self.known_face_names.append(name)  
                else:
                    print(f"No faces found in {filename}. Skipping this image.")

    def register_new_face(self, name: str, image_path: str) -> None:  
        image = face_recognition.load_image_file(image_path)  
        encoding = face_recognition.face_encodings(image)[0]  
        self.known_face_encodings.append(encoding)  
        self.known_face_names.append(name) 

    def get_largest_face_locations(self, frame: np.ndarray) -> Optional[Tuple[int, int, int, int]]:  
        resized_frame = cv2.resize(frame, (0, 0), fx=1/RESCALE_FACTOR, fy=1/RESCALE_FACTOR)  
        rgb_resized_frame = resized_frame[:, :, ::-1]  
        face_locations = face_recognition.face_locations(rgb_resized_frame)  
        if not face_locations:  
            return None  
        if len(face_locations) == 1:  
            return face_locations[0]  
        largest_face_location = max(face_locations, key=lambda loc: (loc[2] - loc[0]) * (loc[1] - loc[3]))  
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

    def run_register(self) -> None:  
        fps_counter = 0
        start_time = time.time()
        fps = 0.0  # Initialize fps variable

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

            face_location = self.get_largest_face_locations(frame)  
            if face_location:  
                top, right, bottom, left = face_location  
                top *= RESCALE_FACTOR  
                right *= RESCALE_FACTOR  
                bottom *= RESCALE_FACTOR  
                left *= RESCALE_FACTOR  
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)  

            combined = cv2.addWeighted(frame, 0.7, mask, 0.3, 0)  

            # Update FPS
            fps_counter += 1
            elapsed_time = time.time() - start_time
            if elapsed_time >= 1.0:  
                fps = fps_counter / elapsed_time
                start_time = time.time()
                fps_counter = 0
            else:
                fps = fps  # Keep the last FPS value

            # Draw FPS on the combined frame
            cv2.putText(combined, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            cv2.imshow('Video', combined)  

            key = cv2.waitKey(1) & 0xFF
            if key == ord('r'):  
                self.register_face()  
            elif key == ord('q'):  
                break  


    def process_frame(self, frame: np.ndarray) -> None:  
        resized_frame = cv2.resize(frame, (0, 0), fx=1/RESCALE_FACTOR, fy=1/RESCALE_FACTOR)  
        rgb_resized_frame = np.ascontiguousarray(resized_frame[:, :, ::-1])  
        face_locations = face_recognition.face_locations(rgb_resized_frame)  
        self.face_encodings = []
        self.face_names = []
        if face_locations:  
            largest_face_location = max(face_locations, key=lambda loc: (loc[2] - loc[0]) * (loc[1] - loc[3]))
            self.face_locations = [largest_face_location]
            self.face_encodings = face_recognition.face_encodings(rgb_resized_frame, self.face_locations)
            if self.face_encodings:  
                face_encoding = self.face_encodings[0]
                name = "Unknown"
                matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding)  
                if any(matches):  
                    face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)  
                    best_match_index = np.argmin(face_distances)  
                    if matches[best_match_index]:  
                        name = self.known_face_names[best_match_index]  
                self.face_names = [name]  

    def display_results(self, frame: np.ndarray) -> None:  
        for (top, right, bottom, left), name in zip(self.face_locations, self.face_names):  
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
        fps = 0.0  # Initialize fps variable

        while True:  
            ret, frame = self.video_capture.read()  
            frame = cv2.flip(frame, 1)  
            if not ret:  
                break  

            self.frame_count += 1  
            if self.frame_count % self.detection_interval == 0:  
                self.process_frame(frame)  

            self.display_results(frame)  

            # Update FPS
            fps_counter += 1
            elapsed_time = time.time() - start_time
            if elapsed_time >= 1.0:  
                fps = fps_counter / elapsed_time
                start_time = time.time()
                fps_counter = 0
            else:
                fps = fps  # Keep the last FPS value

            cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            if self.is_show:
                cv2.imshow('Video', frame)  

            if cv2.waitKey(int(1000 / FPS_LIMIT)) & 0xFF == ord('q'):  
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
