import cv2
import numpy as np
from datetime import datetime
import threading
from CRACKpre import PreImg
def video_frame():
    rtsp_url = "rtsp://192.168.144.25:8554/main.264"

    cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)

    if not cap.isOpened():
        print("Error: Could not open RTSP stream.")
        print("OpenCV backend:", cap.getBackendName())
        print(cv2.getBuildInformation())
    else:
        print("RTSP stream opened successfully.")
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to read frame.")
                break

            cv2.imshow('RTSP Stream', frame)

            key = cv2.waitKey(1)
            if key & 0xFF == ord('k'):
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                img_path = f"captured_image_{timestamp}.jpg"
                cv2.imwrite(img_path, frame)
                print(f"Image saved: {img_path}")

                img_path = img_path.replace(".jpg", "")

                thread = threading.Thread(target=process_image, args=(img_path,))
                thread.start()

            if key & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

def process_image(img_path):
    try:
        PreImg(img_path)
    except Exception as e:
        print(f"Failed to create PreImg: {e}")

video_frame()