import cv2  
import face_recognition  
import numpy as np  
import os  
from typing import List, Optional, Tuple  
  
RESCALE_FACTOR = 4
KNOWN_FACE_DIR = "known_faces"
  
class FaceRecognitionApp:  
    def __init__(self, camera_index: int = 0, detection_interval: int = 2):  
        """  
        Initialize the FaceRecognitionApp with default parameters.  
        """  
        self.video_capture = cv2.VideoCapture(camera_index)  
        self.known_face_encodings: List[np.ndarray] = []  
        self.known_face_names: List[str] = []  
        self.face_locations: List[Tuple[int, int, int, int]] = []  
        self.face_encodings: List[np.ndarray] = []  
        self.face_names: List[str] = []  
        self.process_this_frame: bool = True  
        self.detection_interval: int = detection_interval  
        self.frame_count: int = 0  
        self.new_face_image: Optional[np.ndarray] = None  
  
    def load_known_faces(self, face_images_dir: str) -> None:  
        """  
        Load known face images from a directory.  
        """  
        for filename in os.listdir(face_images_dir):  
            if filename.endswith(".jpg") or filename.endswith(".png"):  
                image_path = os.path.join(face_images_dir, filename)  
                image = face_recognition.load_image_file(image_path)  
                encoding = face_recognition.face_encodings(image)[0]  
                name = os.path.splitext(filename)[0]  
                self.known_face_encodings.append(encoding)  
                self.known_face_names.append(name)  
  
    def register_new_face(self, name: str, image_path: str) -> None:  
        """  
        Register a new face given a name and image path.  
        """  
        image = face_recognition.load_image_file(image_path)  
        encoding = face_recognition.face_encodings(image)[0]  
        self.known_face_encodings.append(encoding)  
        self.known_face_names.append(name)  
  
    def register_face(self) -> None:  
        """  
        Capture a frame from the video and register the face.  
        """  
        ret, frame = self.video_capture.read()  
        if ret:  
            frame = cv2.flip(frame, 1)  # Flip frame horizontally  
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
  
    def get_largest_face_locations(self, frame: np.ndarray) -> Optional[Tuple[int, int, int, int]]:  
        """  
        Get face locations in the given frame.  
        """  
        # Resize frame for faster face recognition processing  
        resized_frame = cv2.resize(frame, (0, 0), fx=1/RESCALE_FACTOR, fy=1/RESCALE_FACTOR)  
        # Convert the image from BGR color (OpenCV) to RGB color (face_recognition)  
        rgb_resized_frame = resized_frame[:, :, ::-1]  
        rgb_resized_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
        # Find all the faces in the current frame  
        face_locations = face_recognition.face_locations(rgb_resized_frame)  
        if not face_locations:  
            return None  
        if len(face_locations) == 1:  
            return face_locations[0]  
        # If multiple faces are detected, return the largest one  
        largest_face_location = max(face_locations, key=lambda loc: (loc[2] - loc[0]) * (loc[1] - loc[3]))  
        return largest_face_location  
  
    def process_frame(self, frame: np.ndarray) -> None:  
        """  
        Process a single frame for face recognition.  
        """  
        # Resize frame for faster face recognition processing  
        resized_frame = cv2.resize(frame, (0, 0), fx=1/RESCALE_FACTOR, fy=1/RESCALE_FACTOR)  
        # Convert the image from BGR color (OpenCV) to RGB color (face_recognition)  
        rgb_resized_frame = resized_frame[:, :, ::-1]  
        rgb_resized_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)  
        # Find all the faces and face encodings in the current frame  
        self.face_locations = face_recognition.face_locations(rgb_resized_frame)  
        self.face_encodings = face_recognition.face_encodings(rgb_resized_frame, self.face_locations)  
        self.face_names = []  
        for face_encoding in self.face_encodings:  
            # See if the face is a match for the known face(s)  
            matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding, tolerance=0.3)  
            name = "Unknown"  
            # Use the known face with the smallest distance to the new face  
            face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)  
            best_match_index = np.argmin(face_distances)  
            if matches[best_match_index]:  
                name = self.known_face_names[best_match_index]  
            self.face_names.append(name)  
  
    def display_results(self, frame: np.ndarray) -> None:  
        """  
        Display the results of face recognition on the frame.  
        """  
        for (top, right, bottom, left), name in zip(self.face_locations, self.face_names):  
            # Scale back up face locations since the frame was scaled down  
            top *= RESCALE_FACTOR  
            right *= RESCALE_FACTOR  
            bottom *= RESCALE_FACTOR  
            left *= RESCALE_FACTOR  
            # Draw a box around the face  
            color = (0, 0, 255) if name == "Unknown" else (255, 0, 0)
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)  
            # Draw a label with a name below the face  
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), color, cv2.FILLED)  
            font = cv2.FONT_HERSHEY_DUPLEX  
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)  
  
    def run_recognition(self) -> None:  
        """  
        Run the face recognition loop.  
        """  
        while True:  
            # Grab a single frame of video  
            ret, frame = self.video_capture.read()  
            frame = cv2.flip(frame, 1)  # Flip frame horizontally  
            if not ret:  
                print("Error: Could not read frame.")  
                break  
            self.frame_count += 1  
            # Only process every 'detection_interval' frames to save time  
            if self.frame_count % self.detection_interval == 0:  
                self.process_frame(frame)  
            # Display the results  
            self.display_results(frame)  
            # Show the resulting image  
            cv2.imshow('Video', frame)  
            # Hit 'q' on the keyboard to quit  
            if cv2.waitKey(1) & 0xFF == ord('q'):  
                break  
  
        # Release handle to the webcam  
        self.video_capture.release()  
        cv2.destroyAllWindows()  
  
    def run_register(self) -> None:  
        """  
        Run the face registration loop.  
        """  
        while True:  
            # Grab a single frame of video  
            ret, frame = self.video_capture.read()  
            frame = cv2.flip(frame, 1)  # Flip frame horizontally  
            if not ret:  
                print("Error: Could not read frame.")  
                break  
  
            # Get the frame dimensions  
            height, width = frame.shape[:2]  
  
            # Define the center rectangle  
            box_width, box_height = 400, 550  # You can adjust the size of the box here  
            left = (width - box_width) // 2  
            top = (height - box_height) // 2  
            right = left + box_width  
            bottom = top + box_height  
  
            mask_color = (200, 200, 200)  
  
            # Create a mask with a transparent rectangle in the center  
            mask = np.zeros_like(frame, dtype=np.uint8)  
            mask[:] = mask_color  # Gray color  
            mask[top:bottom, left:right] = frame[top:bottom, left:right]  
  
            # Draw the center rectangle  
            cv2.rectangle(frame, (left, top), (right, bottom), mask_color, 1)  
  
            # Get the largest face location  
            face_location = self.get_largest_face_locations(frame)  
  
            if face_location:  
                top, right, bottom, left = face_location  
                top *= RESCALE_FACTOR  
                right *= RESCALE_FACTOR  
                bottom *= RESCALE_FACTOR  
                left *= RESCALE_FACTOR  
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)  
  
            # Combine the frame with the mask  
            combined = cv2.addWeighted(frame, 0.7, mask, 0.3, 0)  
  
            # Display the frame with the mask  
            cv2.imshow('Video', combined)  
  
            # Hit 'r' on the keyboard to register new face  
            if cv2.waitKey(1) & 0xFF == ord('r'):  
                self.register_face()  
            # Hit 'q' on the keyboard to quit  
            elif cv2.waitKey(1) & 0xFF == ord('q'):  
                break  
  
        # Release handle to the webcam  
        self.video_capture.release()  
        cv2.destroyAllWindows()  
  
  
if __name__ == "__main__":  
    register_mode = True  # Can be True for registration mode or False for recognition mode  
    app = FaceRecognitionApp()  
    app.load_known_faces('known_faces')  # Load faces from a directory  
    print("Press 'r' to register new face")  
    if register_mode:  
        app.run_register()  # Run registration mode  
    else:  
        app.run_recognition()  # Run recognition mode  
