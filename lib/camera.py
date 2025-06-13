import threading
from threading import Lock
import cv2
import time
import numpy as np
from picamera2 import Picamera2

class PiCamera:
    def __init__(self, fps=20.0):
        self.fps = fps
        self.duration = 1.0 / self.fps

        self.picam = Picamera2()
        config = self.picam.create_still_configuration(
            main={"format": 'RGB888', "size": (640, 480)},
            queue=False)
        self.picam.configure(config)
        self.picam.video_configuration.controls.FrameRate = self.fps
        self.picam.set_controls({"ExposureTime": 8000, "AnalogueGain": 2.5})
        self.picam.start()

        self.fc = None
        self.fc_lock = Lock()

        self.fc_k = np.array([[273.0, 0, 320],
                              [0, 363.8, 240],
                              [0, 0, 1]])
        self.fc_C = np.array([0, 0, 0.18])
        self.fc_R,_  = cv2.Rodrigues(np.array([120*np.pi/180,0,0]))
        self.fc_kinv = np.linalg.inv(self.fc_k)
        self.fc_Rinv = np.linalg.inv(self.fc_R)
    
    def _loop(self): #指定fpsでpicamから画像を取得
        while True:
            current_time = time.time()
            with self.fc_lock:
                self.fc = self.picam.capture_array()
            elapsed = current_time - self.prev_time
            if elapsed < self.duration:
                time.sleep(self.duration - elapsed)
            self.prev_time = time.time()
    
    def get_front_camera(self):
        with self.fc_lock:
            if self.fc is not None:
                return self.fc.copy()
            return None
    
    def fc_convert_2dpos_to_3d(self, point):
        pc = np.array([point[0], point[1], 1.0])
        x_c = self.fc_kinv @ pc
        d_w = self.fc_Rinv @ x_c  # ワールド系での視線ベクトル

        s = -self.fc_C[2] / d_w[2]            # Z=0との交点スケール
        P = self.fc_C + s * d_w               # ワールド座標上の交点
        delta = P - self.fc_C

        angle = np.arctan2(delta[0], delta[1]) * 180 / np.pi  # XY平面上の角度（度）
        return angle, P

    def run(self):
        print(f"PICAMERA start")
        threading.Thread(target=self._loop, daemon=True).start()

class USBCamera:
    def __init__(self, fps=30, width=320, height=240):
        self.fps = fps
        self.duration = 1.0 / self.fps

        self.width = width
        self.height = height
        self.cap = None
        while self.cap is None:
            try:
                self.cap = cv2.VideoCapture(0)
            except Exception as e:
                print(f"Failed to initialize USB camera: {e}")
                time.sleep(1)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M','J','P','G'))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap.set(cv2.CAP_PROP_FPS, self.fps)

        self.lc = np.zeros((self.height, self.width, 3), dtype=np.uint8)  # Initialize with a blank image
        self.lc_lock = Lock()
    
    def _loop(self):
        while True:
            current_time = time.time()
            with self.lc_lock:
                _, self.lc = self.cap.read()
            elapsed = current_time - self.prev_time
            if elapsed < self.duration:
                time.sleep(self.duration - elapsed)
            self.prev_time = time.time()

    def get_line_camera(self):
        with self.lc_lock:
            a = self.lc.copy()
        return a
        
    def run(self):
        print(f"USBCAMERA start")
        threading.Thread(target=self._loop, daemon=True).start()