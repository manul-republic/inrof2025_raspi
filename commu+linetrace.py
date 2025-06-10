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

#速度： -50~50
#ターンスピード： 10～15 chous

framewidth = 320
frameheight = framewidth * 3 // 4  # 4:3のアスペクト比

class USBCamera:
    def __init__(self, width=framewidth, height=frameheight):
        self.width = width
        self.height = height
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M','J','P','G'))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

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
        print(f"CAMERA start")
        threading.Thread(target=self._loop, daemon=True).start()

class LineTracer:
    def __init__(self, slave, camera, angle_threshold=3000, position_threshold=200, frame_check_count=500, debug_stream_enabled=False): # Added debug_stream_enabled
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
    
    # 横棒の数は常に監視しておく
    def _loop(self):
        if self.debug_stream_enabled: # Conditional call
            threading.Thread(target=run_webserver, daemon=True).start()
        loop_start = time.time()
        while True:
            img = self.camera.get_line_camera()
            #debug_img = None
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray,(3,3),0)
            gray = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 5)
            #debug_img = gray.copy()
            debug_img = img.copy()

            self.detect_vertical_line(gray, debug_img)
            self.detect_horizontal_line(gray, debug_img)
            self.command()
            if self.debug_stream_enabled and debug_img is not None: # Conditional call
                update_debug_frame(debug_img)

            # デバッグ出力: ループ時間とライン数
            loop_time = (time.time() - loop_start) * 1000  # ms
            loop_start = time.time()
            #これまでにIDを取得した横棒の数

            time.sleep(0.012)
    def detect_vertical_line(self, gray, debug_img=None):
        size = (int(self.camera.width * 0.8), 20)
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
                if abs(angle_deg) >= 50 and abs(angle_deg) <= 130:
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
        size = (20, int(self.camera.height * 0.4))
        centers = [
            (int(self.camera.width * 0.3), self.camera.height // 4),
            (int(self.camera.width * 0.7), self.camera.height // 4)
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
        print(lines)
    
    def set_command_velocity(self, x, yaw):
        self.slave.set_data(0x03, x)
        self.slave.set_data(0x07, yaw)

    def get_window(self, src, center, size, debug=None): #h, w
        if debug is not None:
            cv2.rectangle(debug, (center[0]-size[0]//2, center[1]-size[1]//2),
                          (center[0]-size[0]//2+size[0], center[1]-size[1]//2+size[1]), (255,0,0))
        return src[center[1]-size[1]//2:center[1]-size[1]//2+size[1],
                   center[0]-size[0]//2:center[0]-size[0]//2+size[0]]
    
    def set_enable(self, is_enabled):
        self.enabled = is_enabled

    def calculate_angular_velocity(self, avg_angle, target_angle=90, kp=-1.0):
        """
        Calculate angular velocity (w) to adjust the average angle towards the target angle.

        Parameters:
        avg_angle (float): The current average angle in degrees.
        target_angle (float): The desired target angle in degrees (default is 90).
        kp (float): Proportional gain for the control system.

        Returns:
        float: Calculated angular velocity (w).
        """
        # Adjust target angle based on the x-coordinate of the vertical line
        if self.vline_current is not None:
            x_offset = self.vline_current[0] - (framewidth / 2)  # Calculate offset from center
            target_angle += -x_offset * 0.2  # Adjust target angle proportionally to the offset
            #print(f"[DEBUG] Adjusted target angle based on x-offset: {target_angle:.2f} degrees")

        angle_error = target_angle - avg_angle
        if abs(angle_error) > 5:
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
        if self.hlines_crossed_count <= 2:
            max_linear_speed = 50
            if not(w == 0):
                v = 0.0
            else:
                v = max_linear_speed
        else:
            v = 0  # Stop moving forward if horizontal line count exceeds 5
            sleep_time = 0.5
            print(f"[DEBUG] Horizontal line count exceeded, stopping forward movement. Waiting for {sleep_time:.2f} seconds.")
            time.sleep(sleep_time)
            self.slave.set_data(0x01, False)
            

        #print(f"[DEBUG] Linear velocity (v): {v:.2f}, Angular velocity (w): {w:.2f}")
        self.set_command_velocity(v, w)
        """
        max_linear_speed = 30
        max_angular_speed = 10
        print(self.vline_current)
        angle = self.vline_current[2]
        if self.vline_current:
            distance_to_line = self.point_to_line_distance(320, 240, self.vline_current)
            if distance_to_line < 30 or :
                v = max_linear_speed
                w = 0.0
            elif:
                v = 0.0
                w = -max_angular_speed if distance_to_line > 0 else max_angular_speed
            else
"""

        #if self.point_to_line_distance(320, 240, self.vline_current) > 30 or 
            

    def run(self):
        print(f"LineTracer start")
        threading.Thread(target=self._loop, daemon=True).start()


if __name__ == "__main__":
    #threading.Thread(target=run_webserver, daemon=True).start()
    slave = SlaveUART(port="/dev/ttyAMA0")  # 使用するシリアルポートを指定
    slave.run()
    cam = USBCamera(width=320, height=240)
    cam.run()
    lt = LineTracer(slave, cam, debug_stream_enabled=False) # Pass the new parameter, True by default
    lt.run()
    time.sleep(1)
    #slave.set_data(0x0d, 30) #yaw
    #slave.set_data(0x0c, 30) #pitch1
    #slave.set_data(0x0b, 0) #pitch2
    slave.set_data(0x01, True)
    time.sleep(1)
    #print(slave.get_data(0x12))
    #slave.set_data(0x01, True)
    try:
        while True:
            time.sleep(0.01)

    except KeyboardInterrupt:
        slave.serial_port.close()
        print("終了")