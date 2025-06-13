import serial
import threading
from threading import Lock
import queue
import time
from struct import pack, unpack
from picamera2 import Picamera2
import cv2
from stream import update_debug_frame, run_webserver

import numpy as np
import math
import uuid
import os
from lib.slave_uart import SlaveUART

from lib.obj_det import ObjectDetector
from lib.slave_uart import SlaveUART

#速度： -50~50
#ターンスピード： 10～15 chous

framewidth = 320
frameheight = framewidth * 3 // 4  # 4:3のアスペクト比
currentmode = 0

# register map
WALK_ENABLE = 0x01
SERVO_ENABLE = 0x02
OBJ_SPEED = 0x03
TURN_OBJ_SPEED = 0x07
ARM_YAW_ANGLE = 0x0b
ARM_PITCH1_ANGLE = 0x0c
ARM_PITCH2_ANGLE = 0x0d
HAND_ANGLE = 0x0e
SUCTION_REF = 0x0f


class PiCamera:
    def __init__(self):
        #self.cap = cv2.VideoCapture(0)
        #self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M','J','P','G'))
        #self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640) # 640x480

        self.picam = Picamera2()
        config = self.picam.create_still_configuration(
            main={"format": 'RGB888', "size": (640, 480)},
            queue=False)
        self.picam.configure(config)
        self.picam.set_controls({"ExposureTime": 8000, "AnalogueGain": 2.5})
        self.picam.start()

        #self.lc = None
        self.fc = None
        #self.lc_lock = Lock()
        self.fc_lock = Lock()

        self.fc_k = np.array([[273.0, 0, 320],
                              [0, 363.8, 240],
                              [0, 0, 1]])
        self.fc_C = np.array([0, 0, 0.18])
        self.fc_R,_  = cv2.Rodrigues(np.array([120*np.pi/180,0,0]))
        self.fc_kinv = np.linalg.inv(self.fc_k)
        self.fc_Rinv = np.linalg.inv(self.fc_R)
    
    def _loop(self):
        while True:
            with self.fc_lock:
                self.fc = self.picam.capture_array()
            time.sleep(0.05)

    
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
    def __init__(self, width=framewidth, height=frameheight):
        self.width = width
        self.height = height
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M','J','P','G'))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap.set(cv2.CAP_PROP_FPS,30)

        """self.picam = Picamera2()
        config = self.picam.create_still_configuration(
            main={"format": 'RGB888', "size": (self.width, self.height)},
            queue=False)
        self.picam.configure(config)
        self.picam.set_controls({"ExposureTime": 8000, "AnalogueGain": 2.5})
        self.picam.start()"""

        self.lc = np.zeros((self.height, self.width, 3), dtype=np.uint8)  # Initialize with a blank image
        self.fc = np.zeros((self.height, self.width, 3), dtype=np.uint8)  # Initialize with a blank image
        self.lc_lock = Lock()
        self.fc_lock = Lock()
    
    def _loop(self):
        while True:
            """with self.fc_lock:
                self.fc = self.picam.capture_array()"""
            with self.lc_lock:
                _, self.lc = self.cap.read()
            time.sleep(0.01)

    def get_line_camera(self):
        with self.lc_lock:
            a = self.lc.copy()
        return a
    
    def get_front_camera(self):
        with self.fc_lock:
            a = self.fc.copy()
        return a
    
    def run(self):
        print(f"USBCAMERA start")
        threading.Thread(target=self._loop, daemon=True).start()

class LineTracer:
    def __init__(
        self, slave, camera, 
        angle_threshold=3000, 
        position_threshold=100, 
        frame_check_count=20, 
        debug_stream_enabled=False,
        mode="forward",
        end_condition="hline_count",
        start_hline_count=0,
        end_hline_count=5,
        end_time=1.0): # Added debug_stream_enabled
        self.camera = camera
        self.slave = slave
        self.debug_stream_enabled = debug_stream_enabled # Initialize debug_stream_enabled

        self.enabled = True
        self.detected_vlines = []
        self.detected_hlines = []
        self.vline_current = None
        self.hlines_current = {}
        self.hlines_crossed_count = 0

        self.angle_threshold = angle_threshold  # Angle difference threshold in degrees
        self.position_threshold = position_threshold  # Position difference threshold in pixels
        self.frame_check_count = frame_check_count  # Number of frames to check for line identity

        self.mode = "forward" # forward, backward
        self.tasks = ["linetrace", "count_vlines"] # "linetrace", "count_vlines"
        self.behavior = "goahead" #goahead, turn, stop

        self.lsd = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD)

        self.mode = mode
        self.end_condition = end_condition
        self.start_hline_count = start_hline_count
        self.end_hilne_count = end_hline_count
        self.end_time = end_time
        self.end = False
        self.walk_stopped = False
        self.proceeding = False
    
    # 横棒の数は常に監視しておく
    def _loop(self):
        if self.debug_stream_enabled: # Conditional call
            threading.Thread(target=run_webserver, daemon=True).start()
        loop_start = time.time()
        proceed_time = 0.0
        while True:
            img = self.camera.get_line_camera()
            #debug_img = None
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray,(3,3),0)
            gray = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 10)
            #debug_img = gray.copy()
            debug_img = img.copy()

            self.detect_vertical_line(gray, debug_img)
            self.detect_horizontal_line(gray, debug_img)
            self.command()
            if self.debug_stream_enabled and debug_img is not None: # Conditional call
                update_debug_frame(debug_img)

            # デバッグ出力: ループ時間とライン数
            if self.proceeding:
                proceed_time += (time.time() - loop_start) * 1000  # ms
            loop_time = (time.time() - loop_start) * 1000  # ms
            loop_start = time.time()
            #これまでにIDを取得した横棒の数

            if (self.end_condition == "time" and proceed_time >= self.end_time):
                self.end = True
            elif (self.end_condition == "hline_count" and self.hlines_crossed_count >= abs(self.end_hilne_count - self.start_hline_count)):
                self.end = True
            
            if self.walk_stopped:
                break
            
            time.sleep(0.012)

    def detect_vertical_line(self, gray, debug_img=None):
        size = (int(self.camera.width * 0.8), 20)
        # if self.slave.get_data(0x00) == (2,):
        if self.mode == "backward":
            centers = [
                (int(self.camera.width * 1.0 // 2), int(self.camera.height * 0.9)),
                (int(self.camera.width * 1.0 // 2), int(self.camera.height * 0.7)),
                (int(self.camera.width * 1.0 // 2), int(self.camera.height * 0.5))
            ]
        else:
            centers = [
                (int(self.camera.width * 1.0 // 2), int(self.camera.height * 0.1)),
                (int(self.camera.width * 1.0 // 2), int(self.camera.height * 0.3)),
                (int(self.camera.width * 1.0 // 2), int(self.camera.height * 0.5))
            ]

        line_perwindow = []
        for i, center in enumerate(centers):
            w = self.get_window(gray, center, size, debug_img)
            contours, _ = cv2.findContours(255 - w, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contours = sorted(contours, key=cv2.contourArea, reverse=True)
            contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 30]

            lines = []
            for ctr in contours:
                line = cv2.fitLine(ctr, cv2.DIST_L2, 0, 0.01, 0.01)
                vx, vy, x, y = line.flatten()
                angle = np.arctan2(vy, vx)
                angle_deg = np.degrees(angle)
                if abs(angle_deg) >= 45 and abs(angle_deg) <= 135:
                    if angle_deg < 0:
                        angle += np.pi
                    lines.append([x + center[0] - size[0] / 2, y + center[1] - size[1] / 2, angle])
            if lines:
                lines = np.array(lines)
                line = np.mean(lines, axis=0)
            else:
                line = None
            line_perwindow.append(line)

        valid_lines = []

        for i in range(len(line_perwindow)):
            for j in range(i + 1, len(line_perwindow)):
                l1 = line_perwindow[i]
                l2 = line_perwindow[j]
                if l1 is not None and l2 is not None:
                    angle_diff = self.angle_difference_deg(l1[2], l2[2])
                    dist = np.linalg.norm(np.array(l1[:2]) - np.array(l2[:2]))
                    if angle_diff < 10 and dist < 200:
                        valid_lines.append(l1)
                        valid_lines.append(l2)

        line_dict = []

        if valid_lines:
            # 重複除去
            unique_valid_lines = [list(x) for x in {tuple(x) for x in valid_lines}]

            # 平均を計算
            avg_line = np.mean(unique_valid_lines, axis=0)
            self.vline_current = avg_line  # Assign the average line to vline_current
            #print(f"[DEBUG] Average Vertical Line: x={avg_line[0]:.2f}, y={avg_line[1]:.2f}, angle={np.degrees(avg_line[2]):.2f} degrees")

            for valid_line in unique_valid_lines:
                identified = False
                for vlinelists in self.detected_vlines[-self.frame_check_count:]:
                    for vlinedict in vlinelists:
                        if self.check_vline_identity(valid_line, vlinedict["line"]):
                            identified = True
                            break
                    if identified:
                        break
                if not identified:
                    id = uuid.uuid1()
                    line_dict.append({"line": valid_line, "id": id})
                    if debug_img is not None:
                        self.debug_draw_line(np.array([valid_line]), debug_img)
                        cv2.putText(debug_img, 'vline detect', (0, 100),
                                    cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2, cv2.LINE_AA)

            # 平均ラインをデバッグ描画
            if debug_img is not None:
                self.debug_draw_line(np.array([avg_line]), debug_img)
                cv2.putText(debug_img, 'avg vline', (0, 150),
                            cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2, cv2.LINE_AA)
        else:
            self.vline_current = None  # Reset vline_current if no valid lines are found
            #print("[DEBUG] No valid vertical lines detected, resetting vline_current.")
        
        self.detected_vlines.append(line_dict)

        if len(self.detected_vlines) > 50:
            self.detected_vlines.pop(0)

        # デバッグ出力: 認識した縦線の傾きと座標
        #for line in valid_lines:
        #    print(f"[DEBUG] Vertical line detected: x={line[0]:.2f}, y={line[1]:.2f}, angle={np.degrees(line[2]):.2f} degrees")

    
    def detect_horizontal_line(self, gray, debug_img=None):
        # currentmode = self.slave.get_data(0x00)
        # if currentmode == (0,):
        if self.mode == "forward":
            centerheight = self.camera.height // 4
        # elif currentmode == (2,) and self.hlines_crossed_count == 0:
        elif self.mode == "backward" and self.hlines_crossed_count == 0:
            centerheight = self.camera.height * 3 // 4
        else:
            centerheight = self.camera.height // 4
        size = (20, int(self.camera.height * 0.5))
        centers = [
            (int(self.camera.width * 0.35), centerheight),
            (int(self.camera.width * 0.65), centerheight)
        ]
        est = []
        detected = False
        for center in centers:
            w = self.get_window(gray, center, size, debug_img)
            contours, _ = cv2.findContours(255-w, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contours = sorted(contours, key=cv2.contourArea, reverse=True)
            contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 30]

            if len(contours) >= 2:
                lines = []
                for ctr in contours:
                    line = cv2.fitLine(ctr, cv2.DIST_L2, 0, 0.01, 0.01)
                    vx, vy, x, y = line.flatten()
                    angle = np.arctan2(vy, vx)  # angle in radians
                    lines.append([x+center[0]-size[0]/2, y+center[1]-size[1]/2, angle])
                lines = np.array(lines)
                #改良の余地あり: ペアを確認
                line = np.mean(lines, axis=0)
                est.append(line)
            # デバッグ出力: 各ウィンドウで認識したライン
            #print(f"[DEBUG][hline] window center={center} line={line if len(contours) >= 2 else None}")
        
        #idはlistのindexとし、idを取得
        line_dict = []
        if len(est) == 2:
            l1 = est[0]; l2 = est[1]
            if self.angle_difference_deg(l1[2], l2[2]) < 15:
                x0, y0 = l1[:2]   
                if self.point_to_line_distance(x0, y0, l2) < 20:
                    detected = True
                    #lineの同一性評価をしたい
                    id = self.get_hline_id(l1)
                    ldict = {"line": l1, "id":id, }
                    line_dict.append(ldict)
                    if debug_img is not None:
                        self.debug_draw_line(line, debug_img)
                        cv2.putText(debug_img, 'hline detect', (0, 50), cv2.FONT_HERSHEY_PLAIN, 4, (255, 255, 255), 5, cv2.LINE_AA)

        self.detected_hlines.append(line_dict)

        if len(self.detected_hlines) > 50:
            self.detected_hlines.pop(0)
        
        self.hlines_current = {}

        # Set memory[1] to False if two horizontal lines are detected
        #if len(est) == 2:
            #print("[DEBUG] Two horizontal lines detected. Setting memory[1] to False.")

    #過去10フレームに検出したlineで近しいものがあればidを流用
    def get_hline_id(self, line):
        for hlinedicts in self.detected_hlines[-self.frame_check_count:]:
            for hlinedict in hlinedicts:
                hline = hlinedict["line"]
                if self.check_hline_identity(line, hline):
                    return hlinedict['id']
        
        # Increment the count for uniquely recognized horizontal lines
        self.hlines_crossed_count += 1
        print(f"[DEBUG] Total unique horizontal lines counted: {self.hlines_crossed_count}")
        
        return uuid.uuid1()

    def debug_draw_line(self, lines, img):
        if img is None: return
        _, cols = img.shape[:2]
        _lines = lines.reshape([-1, 3])
        for line in _lines:
            x, y, angle = line
            length = 200  # length of the line to draw
            x1 = int(x - length * np.cos(angle) / 2)
            y1 = int(y - length * np.sin(angle) / 2)
            x2 = int(x + length * np.cos(angle) / 2)
            y2 = int(y + length * np.sin(angle) / 2)
            cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
            #print(f"[DEBUG] Drawing line with angle: {np.degrees(angle):.2f} degrees")
    
    def angle_difference_deg(self, a1, a2):
        diff = abs(a1 - a2) % 180
        return min(diff, 180 - diff)  # 方向が反対でもOK

    def point_to_line_distance(self, x0, y0, line):
        x1, y1, angle_deg = line    #点(x0, y0) から line=(x, y, angle) に引かれた直線までの距離
        angle_rad = np.radians(angle_deg)
        return abs((x0 - x1)*np.sin(angle_rad) - (y0 - y1)*np.cos(angle_rad))   # 法線ベクトル (sinθ, -cosθ) による点と直線の距離

    def check_hline_identity(self, l1, l2):
        # Adjusted to use class thresholds for angle and position
        if math.isclose(l1[0], l2[0], abs_tol=self.position_threshold) and \
           math.isclose(l1[1], l2[1], abs_tol=self.position_threshold) and \
           self.angle_difference_deg(l1[2], l2[2]) < math.radians(self.angle_threshold):
            return True
        return False
    
    def check_vline_identity(self, l1, l2):
        # Vertical line identity check is disabled
        return False

    def get_line_distance(self, l1, l2):
        p = (l2[0:2] + l2[2:4]) / 2
        ap = p - l1[0:2]
        ab = l1[2:4] - l1[0:2]
        ba = l1[0:2] - l1[2:4]
        bp = p - l1[2:4]
        ai_norm = np.dot(ap, ab)/np.linalg.norm(ab)
        neighbor_point = l1[0:2] + (ab)/np.linalg.norm(ab)*ai_norm
        return np.linalg.norm(p - neighbor_point)

    def get_line_dir_in_window(self, window, length_thre=25):
        lines, width, prec, nfa = self.lsd.detect(window)
        #長さが一定以上のlineだけ残す
        length = np.linalg.norm(lines[:,0,0:2]-lines[:,0,2:4])
        lines = lines[length > length_thre, :]
    
    def set_command_velocity(self, x, yaw):
        self.slave.set_data(OBJ_SPEED, x)
        self.slave.set_data(TURN_OBJ_SPEED, yaw)

    def get_window(self, src, center, size, debug=None): #h, w
        if debug is not None:
            cv2.rectangle(debug, (center[0]-size[0]//2, center[1]-size[1]//2),
                          (center[0]-size[0]//2+size[0], center[1]-size[1]//2+size[1]), (255,0,0))
        return src[center[1]-size[1]//2:center[1]-size[1]//2+size[1],
                   center[0]-size[0]//2:center[0]-size[0]//2+size[0]]
    
    def set_enable(self, is_enabled):
        self.enabled = is_enabled

    def calculate_angular_velocity(self, avg_angle, target_angle=90, kp=-1.0):
        # Adjust target angle based on the x-coordinate of the vertical line
        if self.vline_current is not None:
            x_offset = self.vline_current[0] - (framewidth / 2)  # Calculate offset from center
            # if self.slave.get_data(0x00) == (2,):
            if self.mode == "backward":
                target_angle += x_offset * 0.2  # Adjust target angle proportionally to the offset
            else:
                target_angle -= x_offset * 0.2
            #print(f"[DEBUG] Adjusted target angle based on x-offset: {target_angle:.2f} degrees")

        angle_error = target_angle - avg_angle
        if abs(angle_error) > 7:
            w = kp * angle_error
        else:
            w = 0.0
        return w

    def command(self):
        if self.vline_current is not None and len(self.vline_current) == 3:
            avg_angle_deg = np.degrees(self.vline_current[2])
            w = self.calculate_angular_velocity(avg_angle_deg)
            #print(f"[DEBUG] Calculated angular velocity (w): {w:.2f}")
        else:
            w = 0.0
            #print("[DEBUG] No valid vertical line detected, setting angular velocity (w) to 0.")

        # Adjust linear velocity (v) based on horizontal line count and magnitude of w
        # print(ballcount)
        # if ballcount == 0:
        #     LineToGo = 4
        # else:
        #     LineToGo = 2
        # if self.hlines_crossed_count <= LineToGo and self.slave.get_data(0x00) == (0,):
        #     if self.hlines_crossed_count <= 1:
        #         max_linear_speed = 20
        #     elif self.hlines_crossed_count > 3:
        #         max_linear_speed = 30
        #     else:
        #         max_linear_speed = 40
        #     if not(w == 0):
        #         v = 0.0
        #     else:
        #         v = max_linear_speed
        # elif self.hlines_crossed_count <= 2 and self.slave.get_data(0x00) == (2,):
        #     max_linear_speed = -20
        #     if not(w == 0):
        #         v = 0.0
        #     else:
        #         v = max_linear_speed
        if self.mode == "forward":
            max_linear_speed = 20
            if not(w == 0):
                v = 0.0
            else:
                v = max_linear_speed
        elif self.mode == "backward":
            max_linear_speed = -20
            if not(w == 0):
                v = 0.0
            else:
                v = max_linear_speed
        self.proceeding = (v != 0)

        if self.end:
            v = 0  # Stop moving forward if horizontal line count exceeds 5
            self.slave.set_data(WALK_ENABLE, False)
            self.set_command_velocity(0, 0)
            sleep_time = 0.5
            print(f"[DEBUG] Horizontal line count exceeded, stopping forward movement. Waiting for {sleep_time:.2f} seconds.")
            time.sleep(sleep_time)
            self.slave.set_data(WALK_ENABLE, False)
            # self.slave.set_data(0x00, 1)
            # currentmode = (1,)
            # while currentmode == (1,):
            #     time.sleep(1)
            #     currentmode = self.slave.get_data(0x00)
            # self.hlines_crossed_count = 0
            self.walk_stopped = True
            

        #print(f"[DEBUG] Linear velocity (v): {v:.2f}, Angular velocity (w): {w:.2f}")
        self.set_command_velocity(v, w)

        #if self.point_to_line_distance(320, 240, self.vline_current) > 30 or 
            

    def run(self):
        print(f"LineTracer start")
        threading.Thread(target=self._loop, daemon=True).start()
        while not self.walk_stopped:
            time.sleep(1)

# TODO
#   色分けの反映
#   済 ボールへのにじり寄りを行った場合の退却
#   オブジェクトが見つからなかった場合の諦め
#   オブジェクトを探す場所の変更
#   スタート台への帰還

if __name__ == "__main__":
    #threading.Thread(target=run_webserver, daemon=True).start()
    slave = SlaveUART(port="/dev/ttyAMA0")  # 使用するシリアルポートを指定

    def walk(length, vel=15):
        if length == 0:
            return 0
        elif length > 0:
            vel = abs(vel)
        else:
            vel = -abs(vel)
        slave.set_data(WALK_ENABLE, True)
        slave.set_data(OBJ_SPEED, vel)
        slave.set_data(TURN_OBJ_SPEED, 0)
        time.sleep((length / vel) if vel > 0 else (length / vel / 4))
        slave.set_data(OBJ_SPEED, 0)
        slave.set_data(WALK_ENABLE, False)
        time.sleep(0.5)
        return length

    def turn(angle, omega=15):
        if angle == 0:
            return 0
        elif angle > 0:
            omega = abs(omega)
        if angle < 0:
            omega = -abs(omega)
        slave.set_data(WALK_ENABLE, True)
        slave.set_data(OBJ_SPEED, 0)
        slave.set_data(TURN_OBJ_SPEED, omega)
        time.sleep(angle / omega)
        slave.set_data(TURN_OBJ_SPEED, 0)
        slave.set_data(WALK_ENABLE, False)
        time.sleep(0.5)
        return angle

    slave.run()
    # slave.set_data(0x00, 0)
    
    ballcount = 0
    searched_length = 0

    cam = USBCamera(width=320, height=240)
    cam.run()
    picam = PiCamera()
    picam.run()
    objdet = ObjectDetector("/home/teba/Programs/inrof2025/python/lib/masters.onnx")
    # lt = LineTracer(slave, cam, debug_stream_enabled=False)
    # lt.run()

    time.sleep(0.2)
    slave.set_data(SERVO_ENABLE, True) #servo on
    time.sleep(3)
    slave.set_data(SUCTION_REF, 8.0)
    slave.set_data(ARM_PITCH2_ANGLE, 145)
    #アーム展開
    slave.set_data(ARM_YAW_ANGLE, 40)
    time.sleep(0.5)
    slave.set_data(ARM_PITCH1_ANGLE, 90)
    time.sleep(1)
    # slave.set_data(SUCTION_REF, 0.0)
    slave.set_data(ARM_PITCH1_ANGLE, 180)
    time.sleep(0.3)
    slave.set_data(ARM_YAW_ANGLE, 100)
    time.sleep(1)
    #slave.set_data(ARM_YAW_ANGLE, 105)
    slave.set_data(SUCTION_REF, 0.0)
    #アーム収納
    slave.set_data(ARM_PITCH1_ANGLE, 90)
    time.sleep(1)
    slave.set_data(ARM_YAW_ANGLE, 40)
    time.sleep(0.5)
    slave.set_data(ARM_PITCH1_ANGLE, 0)
    time.sleep(0.3)
    slave.set_data(ARM_PITCH2_ANGLE, 105)
    slave.set_data(ARM_YAW_ANGLE, 0)

    while True:
        # continue
        # slave.set_data(0x00, 0)
        # currentmode = slave.get_data(0x00)
        # slave.set_data(WALK_ENABLE, True)
        # while currentmode == (0,):
        #     time.sleep(0.1)
        #     currentmode = slave.get_data(0x00)
        slave.set_data(WALK_ENABLE, True)
        lt = LineTracer(
            slave, cam, debug_stream_enabled=False, 
            mode="forward", end_condition="hline_count", 
            start_hline_count=(0 if ballcount == 0 else 2), 
            end_hline_count=5)
        lt.run()
        print("linetracefinished")
        walk(searched_length)
        while True:
            valid_object_found = False
            for i in range(2):
                proceed_length = 0
                turn_angle = 0
                turn_angle += turn(15 if i == 0 else -15)
                error_count = 0
                while True:
                    img = picam.get_front_camera()
                    if img is None:
                        time.sleep(0.1)
                        print("cannot aquire camera image!")
                        continue
                    outputs = objdet.predict(img)
                    print("outputs:", outputs)
                    if outputs is not None:
                        # スコアとクラス
                        bboxes = outputs[:, 0:4]
                        cls = outputs[:, 6]
                        scores = outputs[:, 4] * outputs[:, 5]

                        # 各クラスごとのbbox抽出
                        redballs = bboxes[(cls == 0) & (scores > 0.6)]
                        blueballs = bboxes[(cls == 1) & (scores > 0.6)]
                        yellowcans = bboxes[(cls == 3) & (scores > 0.6)]

                        # 面積を計算
                        def calc_areas(boxes):
                            return np.abs(boxes[:, 2] - boxes[:, 0]) * np.abs(boxes[:, 3] - boxes[:, 1])

                        red_areas = calc_areas(redballs)
                        blue_areas = calc_areas(blueballs)
                        yellow_areas = calc_areas(yellowcans)

                        # 全データ結合（bbox, area, class_id）
                        all_objects = []
                        for boxes, areas, label in [(redballs, red_areas, 0), (blueballs, blue_areas, 1), (yellowcans, yellow_areas, 2)]:
                            for i in range(len(boxes)):
                                all_objects.append((boxes[i], areas[i], label))

                        # 面積最大のオブジェクトを選択
                        if all_objects:
                            max_obj = max(all_objects, key=lambda x: x[1])
                            bbox, area, label = max_obj
                            center = [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2]
                            #print(f"最大面積オブジェクト（クラス{label}）: 2D中心={center}, 面積={area}")
                            objecttheta,objectposition = picam.fc_convert_2dpos_to_3d(center)
                            objectdist = objectposition[1]**2+objectposition[0]**2+objectposition[2]**2
                            print(f"最大面積オブジェクト（クラス{label}）: 2D中心={center}, 面積={area}, 角度={objecttheta:.2f}, 距離={objectdist:.2f}m")
                            if objectdist < 0.05:
                                slave.set_data(ARM_PITCH2_ANGLE, int(np.clip(-objecttheta*2/4+105,0,180)))
                                if True:
                                    #アーム展開
                                    slave.set_data(ARM_YAW_ANGLE, 40)
                                    time.sleep(0.5)
                                    slave.set_data(ARM_PITCH1_ANGLE, 90)
                                    time.sleep(1)
                                    slave.set_data(ARM_PITCH1_ANGLE, 180)
                                    time.sleep(0.3)
                                    slave.set_data(ARM_YAW_ANGLE, 80)
                                    time.sleep(1)
                                    slave.set_data(SUCTION_REF, 0.99)
                                    time.sleep(1)
                                    slave.set_data(ARM_YAW_ANGLE, 100)
                                    time.sleep(0.5)
                                    slave.set_data(ARM_YAW_ANGLE, 90)
                                    time.sleep(0.5)
                                    #アーム収納
                                    slave.set_data(ARM_PITCH1_ANGLE, 90)
                                    time.sleep(1)
                                    slave.set_data(ARM_YAW_ANGLE, 40)
                                    time.sleep(0.5)
                                    slave.set_data(ARM_PITCH1_ANGLE, 0)
                                    time.sleep(0.3)
                                    slave.set_data(ARM_PITCH2_ANGLE, 105)
                                    slave.set_data(ARM_YAW_ANGLE, 0)
                                    time.sleep(1.5)
                                    slave.set_data(SUCTION_REF, 0.7)
                                    valid_object_found = True
                                    break
                            else:
                                proceed_length += walk(22.5 if objectdist > 0.1 else 15.0)
                                error_count += 1
                        else:
                            print("有効なオブジェクトが見つかりませんでした")
                            error_count += 1
                    else:
                        print("no detection acquired")
                        error_count += 1
                    if error_count > 5:
                        break
                    time.sleep(0.1)

                proceed_length += walk(-proceed_length)
                turn_angle += turn(-turn_angle)
                if valid_object_found:
                    walk(60)
                    break
            if valid_object_found:
                break
            else:
                lt = LineTracer(
                    slave, cam, debug_stream_enabled=False, 
                    mode="forward", end_condition="time", 
                    end_time=1.0)
                lt.run()
                searched_length += 20
                continue
        # slave.set_data(WALK_ENABLE, True)
        # slave.set_data(0x00, 2)
        # currentmode = (2,)
        # while currentmode == (2,):
        #     time.sleep(0.1)
        #     currentmode = slave.get_data(0x00)
        slave.set_data(WALK_ENABLE, True)
        lt = LineTracer(
            slave, cam, debug_stream_enabled=False, 
            mode="backward", end_condition="hline_count", 
            start_hline_count=7, 
            end_hline_count=3)
        lt.run()
        walk(-15)
        slave.set_data(SUCTION_REF, 8.0)
        slave.set_data(ARM_PITCH2_ANGLE, 195)
        #アーム展開
        slave.set_data(ARM_YAW_ANGLE, 40)
        time.sleep(0.5)
        slave.set_data(ARM_PITCH1_ANGLE, 90)
        time.sleep(1)
        slave.set_data(SUCTION_REF, 0.0)
        slave.set_data(ARM_PITCH1_ANGLE, 180)
        time.sleep(0.3)
        slave.set_data(ARM_YAW_ANGLE, 100)
        time.sleep(1)
        #slave.set_data(ARM_YAW_ANGLE, 105)
        slave.set_data(SUCTION_REF, 0.0)
        #アーム収納
        slave.set_data(ARM_PITCH1_ANGLE, 90)
        time.sleep(1)
        slave.set_data(ARM_YAW_ANGLE, 40)
        time.sleep(0.5)
        slave.set_data(ARM_PITCH1_ANGLE, 0)
        time.sleep(0.3)
        slave.set_data(ARM_PITCH2_ANGLE, 105)
        slave.set_data(ARM_YAW_ANGLE, 0)
        ballcount += 1
        print(f"Ball count: {ballcount}")
        

                
                

            
    
    #slave.set_data(ARM_PITCH2_ANGLE, 30) #yaw
    #slave.set_data(ARM_PITCH1_ANGLE, 30) #pitch1
    #slave.set_data(ARM_YAW_ANGLE, 0) #pitch2
    #slave.set_data(WALK_ENABLE, True)
    time.sleep(1)
    #print(slave.get_data(0x12))
    #slave.set_data(WALK_ENABLE, True)
    try:
        while True:
            time.sleep(0.01)

    except KeyboardInterrupt:
        slave.serial_port.close()
        print("終了")