import serial
import threading
from threading import Lock
import queue
import time
from struct import pack, unpack
from picamera2 import Picamera2
import numpy as np
import cv2

from lib.obj_det import ObjectDetector
from lib.slave_uart import SlaveUART

#速度： -50~50
#ターンスピード： 10～15 chous

class PiCamera:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M','J','P','G'))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640) # 640x480

        self.picam = Picamera2()
        config = self.picam.create_still_configuration(
            main={"format": 'RGB888', "size": (640, 480)},
            queue=False)
        self.picam.configure(config)
        self.picam.set_controls({"ExposureTime": 8000, "AnalogueGain": 2.5})
        self.picam.start()

        self.lc = None
        self.fc = None
        self.lc_lock = Lock()
        self.fc_lock = Lock()

        self.fc_k = np.array([[273.0, 0, 320],
                              [0, 363.8, 240],
                              [0, 0, 1]])
        self.fc_C = np.array([0, 0, 0.18])
        self.fc_R = np.array([[1, 0, 0],
                              [0, 0.866, -0.5],
                              [0, 0.5, 866]])
        self.fc_kinv = np.linalg.inv(self.fc_k)
        self.fc_Rinv = np.linalg.inv(self.fc_R)
    
    def _loop(self):
        while True:
            with self.fc_lock:
                self.fc = self.picam.capture_array()
            with self.lc_lock:
                _, self.lc = self.cap.read()
                self.lc = cv2.resize(self.lc, (640, 480))
            time.sleep(0.05)

    def get_line_camera(self):
        with self.lc_lock:
            if self.lc is not None:
                return self.lc.copy()
            return None

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
        print(f"CAMERA start")
        threading.Thread(target=self._loop, daemon=True).start()


if __name__ == "__main__":
    slave = SlaveUART(port="/dev/ttyAMA0")  # 使用するシリアルポートを指定
    slave.run()
    cam = PiCamera()
    cam.run()
    objdet = ObjectDetector("/home/teba/Programs/inrof2025/python/lib/masters.onnx")

    try:
        while True:
            img = cam.get_front_camera()
            if img is None:
                time.sleep(0.1)
                print("cannot aquire camera image!")
                continue
            outputs = objdet.predict(img)
            print("outputs:", outputs)
            if outputs is not None:
                print("classes:", outputs[:, 6])
                print("scores:", outputs[:, 4] * outputs[:, 5])
                bboxes = outputs[:, 0:4]
                cls = outputs[:, 6]
                scores = outputs[:, 4] * outputs[:, 5]
                redballs = bboxes[(cls == 0) & (scores > 0.6), 0:4]

                areas = np.abs(redballs[:, 2] - redballs[:, 0]) * np.abs(redballs[:, 3] - redballs[:, 1])
                #max_index = np.argmax(areas)
                if redballs.size > 0:
                    target = redballs[0]
                    center = [(target[0]+target[2]) /2, (target[1]+target[3]) /2]
                    print(center)
                    h, w = img.shape[:2]   
                    img_center_x = w / 2
                    
                    if abs(center[0] - img_center_x) > 25:
                        if (center[0] - img_center_x) > 0:
                            print("turn right!!")
                            slave.set_data(0x02, True)
                            slave.set_data(0x03, 0.0)
                            slave.set_data(0x07, 10)
                        else:
                            print("turn left!!")
                            slave.set_data(0x02, True)
                            slave.set_data(0x03, 0.0)
                            slave.set_data(0x07, -10)
                    else:
                        print("stop!!!")
                        slave.set_data(0x02, False)
                        slave.set_data(0x03, 0.0)
                        slave.set_data(0x07, 0.0)
                    print(f"object 2dpos:  {center}, estimated 3dpos: {cam.fc_convert_2dpos_to_3d(center)}")

                else:
                    print("there is no redball!")
            else:
                print("no detection acquired")
            time.sleep(0.1)

            
    except KeyboardInterrupt:
        slave.serial_port.close()
        print("終了")