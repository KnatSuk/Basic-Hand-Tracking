from turtle import hideturtle
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import image_classifier
import numpy as np
import cv2
import time


mp_hands = mp.tasks.vision.HandLandmarksConnections
mp_drawing = mp.tasks.vision.drawing_utils
mp_drawing_styles = mp.tasks.vision.drawing_styles

class HandDetector:
    COLORS = {
        "thumb":  (0, 0, 255),      # Red
        "index":  (0, 255, 0),      # Green
        "middle": (255, 255, 0),    # Light blue
        "ring":   (0, 255, 255),    # Yellow
        "pinky":  (255, 0, 255),    # Purple
        "palm":   (255, 0, 0)       # Blue
    }

    FINGER_CONNECTIONS = {
        "thumb":  [(1,2),(2,3),(3,4)],
        "index":  [(5,6),(6,7),(7,8)],
        "middle": [(9,10),(10,11),(11,12)],
        "ring":   [(13,14),(14,15),(15,16)],
        "pinky":  [(17,18),(18,19),(19,20)]
    }

    PALM_CONNECTIONS = [
        (0,1),(1,5),(5,9),(9,13),(13,17),(17,0)
    ]

    def __init__(self):
        self.base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
        self.options = vision.HandLandmarkerOptions(base_options=self.base_options, num_hands=4)
        self.detector = vision.HandLandmarker.create_from_options(self.options)

    def detected_result(self, image):
        detection_result = self.detector.detect(image)
        return detection_result

    def display_result(self, image, detection_result):
        height, width, _ = image.shape
        for hands in detection_result.hand_landmarks:
            self.draw_finger(image, hands, height, width)
            self.draw_palm(image, hands, height, width)
    
    def draw_finger(self, image, hand, height, width):
        for finger, connections in self.FINGER_CONNECTIONS.items():

            for start, end in connections:
                x1 = int(hand[start].x * width)
                y1 = int(hand[start].y * height)
                x2 = int(hand[end].x * width)
                y2 = int(hand[end].y * height)
            
                cv2.line(image, (x1, y1), (x2, y2), self.COLORS[finger], 3)

                cv2.circle(image, (x1, y1), 6, self.COLORS[finger], -1)
                cv2.circle(image, (x2, y2), 6, self.COLORS[finger], -1)
    
    def draw_palm(self, image, hand, height, width):
        for start, end in self.PALM_CONNECTIONS:
            x1 = int(hand[start].x * width)
            y1 = int(hand[start].y * height)
            x2 = int(hand[end].x * width)
            y2 = int(hand[end].y * height)

            cv2.line(image, (x1,y1), (x2,y2), self.COLORS["palm"], 2)
            cv2.circle(image, (x1, y1), 6, self.COLORS["palm"], -1)



def main():
    current_time = 0
    previous_time = 0
    cap = cv2.VideoCapture(1)
    detector = HandDetector()
    print("hwy")

    while True:
        success, img = cap.read()
        if not success:
            break

        rgb_frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        detected_result = detector.detected_result(mp_image)
        detector.display_result(img, detected_result)


        current_time = time.time()
        fps = 1 / (current_time - previous_time)
        previous_time = current_time

        cv2.imshow("Detection", img)
        if cv2.waitKey(1) & 0XFF == ord("q"):
            break
        pass

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()