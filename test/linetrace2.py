import multiprocessing as mp
import cv2
import numpy as np
import math

def main():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M','J','P','G'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    lsd = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD)

    while True:
        ret, img = cap.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        #gaussianの二値化がいい感じ
        # 板の継ぎ目が見えちゃうけど黒線は2つのラインで表現されるからそれ検出できれば無視できる
        #gray = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,63,2)
        gray = cv2.GaussianBlur(gray,(3,3),0)
        gray = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 5)

        ret,gray = cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        #blur = cv2.GaussianBlur(gray,(5,5),0)
        #ret,gray = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

        lines, width, prec, nfa = lsd.detect(gray)
        out = img.copy()
        out = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        if lines is not None:
            #lsd.drawSegments(out, lines)
            length = np.linalg.norm(lines[:,0,0:2] - lines[:,0,2:4], axis=1)
            lines = lines[length > 100, :]
            for idx in range(len(lines)):
                x1, y1, x2, y2 = lines[idx][0] 
                cv2.line(out, (int(x1), int(y1)), (int(x2), int(y2)), (255,0,0), int(width[idx]))
        #cv2.imshow("Test", out)
        #cv2.imshow("test", img)
        if cv2.waitKey(1) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

import serial
import threading
from threading import Lock
import queue
import time
from struct import pack, unpack
from picamera2 import Picamera2
import cv2

import numpy as np
import math
import uuid
import os

class Camera:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M','J','P','G'))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640) # 640x480

        """self.picam = Picamera2()
        config = self.picam.create_still_configuration(
            main={"format": 'RGB888', "size": (640, 480)},
            queue=False)
        self.picam.configure(config)
        self.picam.set_controls({"ExposureTime": 8000, "AnalogueGain": 2.5})
        self.picam.start()"""

        self.lc = None
        self.fc = None
        self.lc_lock = Lock()
        self.fc_lock = Lock()

        self.fps = 30
        self._duration = 1 / self.fps
    
    def __del__(self):
        self.picam.release()


    def _loop(self):
        while True:
            start = time.time()
            """with self.fc_lock:
                self.fc = self.picam.capture_array()"""
            with self.lc_lock:
                _, self.lc = self.cap.read()
            elapsed = time.time() - start
            time_to_sleep = max(0, self._duration - elapsed)
            time.sleep(time_to_sleep)

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
    def __init__(self, slave, camera):
        self.camera = camera
        self.slave = slave

        self.enabled = True
        self.detected_vlines = []
        self.detected_hlines = []
        self.vline_current = None
        self.hlines_current = {}
        self.hlines_crossed_count = 0

        self.mode = "forward" # forward, backward
        self.tasks = ["linetrace", "count_vlines"] # "linetrace", "count_vlines"
        self.behavior = "goahead" #goahead, turn, stop
    
    # 横棒の数は常に監視しておく
    def _loop(self):
        while True:
            img = self.camera.get_line_camera()
            debug_img = img.copy()
            #debug_img = None
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray,(3,3),0)
            gray = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 5)

            self.detect_vertical_line(gray, debug_img)
            self.detect_horizontal_line(gray, debug_img)
            self.command()
            if debug_img is not None:
                #cv2.imshow("test", debug_img)
                cv2.imwrite(os.path.join("/home/teba/Documents/movie", str(uuid.uuid1())+".png"), img)
                cv2.imwrite(os.path.join("/home/teba/Documents/movie", str(uuid.uuid1())+"d.png"), debug_img)

                print("saved")
            time.sleep(0.01)
    
    def detect_vertical_line(self, gray, debug_img=None):
        size = (240,40)
        centers = [(320,120), (320,240), (320,360)]

        line_perwindow = []
        for i, center in enumerate(centers):
            w = self.get_window(gray, center, size, debug_img)
            contours, _ = cv2.findContours(255-w, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contours = sorted(contours, key=cv2.contourArea, reverse=True)
            contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 30]

            lines = []; line = None
            if len(contours) >= 2:
                for ctr in contours:
                    line = cv2.fitLine(ctr, cv2.DIST_L2, 0, 0.01, 0.01)
                    vx, vy, x, y = line.flatten()
                    angle = np.arctan2(vy, vx)  # angle in radians
                    lines.append([x+center[0]-size[0]/2, y+center[1]-size[1]/2, angle])
                lines = np.array(lines)
                #改良の余地あり: ペアを確認
                line = np.mean(lines, axis=0)
                self.debug_draw_line(line, debug_img)
            line_perwindow.append(line)
        
        for i, vline in enumerate(line_perwindow):
            if line_perwindow[i] is None:
                continue
            identified = False
            for vlinelists in self.detected_vlines[-10:]:
                if vlinelists[i]: # None回避
                    if self.check_line_identity(vline, vlinelists[i]):
                        identified = True
                        continue
            if not identified:
                line_perwindow[i] = None
        
        self.detected_vlines.append(line_perwindow)
        
        # Extract elements which are not None and get mean value
        valid_lines = [line for line in line_perwindow if line is not None]
        if valid_lines:
            self.vline_current = np.mean(valid_lines, axis=0)
        else:
            self.vline_current = None
        
        if len(self.detected_vlines) > 50:
            self.detected_vlines.pop(0)


    def detect_horizontal_line(self, gray, debug_img=None):
        size = (40,300)
        centers = [(200,240), (440,240)]
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
        
        #idはlistのindexとし、idを取得
        line_dict = []
        if len(est) == 2:
            l1 = est[0]; l2 = est[1]
            if self.angle_difference_deg(l1[2], l2[2]) < 15:
                x0, y0 = l1[:2]   
                if self.point_to_line_distance(x0, y0, l2) < 10:
                    detected = True
                    if debug_img is not None:
                        self.debug_draw_line(line, debug_img)
                        cv2.putText(debug_img, 'hline detect', (0, 50), cv2.FONT_HERSHEY_PLAIN, 4, (255, 255, 255), 5, cv2.LINE_AA)
                    #lineの同一性評価をしたい
                    id = self.get_hline_id(l1)
                    ldict = {"line": l1, "id":id, }
                    line_dict.append(ldict)

        self.detected_hlines.append(line_dict)

        if len(self.detected_hlines) > 50:
            self.detected_hlines.pop(0)
        
        self.hlines_current = {}

    #過去10フレームに検出したlineで近しいものがあればidを流用
    def get_hline_id(self, line):
        for hlinedicts in self.detected_hlines[-10:]:
            for hlinedict in hlinedicts:
                hline = hlinedict["line"]
                if self.check_line_identity(line, hline):
                    return hlinedict['id']
        self.hlines_crossed_count += 1
        print("horizontal line counted")
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
    
    def angle_difference_deg(self, a1, a2):
        diff = abs(a1 - a2) % 180
        return min(diff, 180 - diff)  # 方向が反対でもOK

    def point_to_line_distance(self, x0, y0, line):
        x1, y1, angle_deg = line    #点(x0, y0) から line=(x, y, angle) に引かれた直線までの距離
        angle_rad = np.radians(angle_deg)
        return abs((x0 - x1)*np.sin(angle_rad) - (y0 - y1)*np.cos(angle_rad))   # 法線ベクトル (sinθ, -cosθ) による点と直線の距離

    def check_line_identity(self, l1, l2):
        # なんか線間距離とangle違いだとうまくいかない x,y,angleの違いで単純に処理
        print(self.angle_difference_deg(l1[2], l2[2]))
        if math.isclose(l1[0], l2[0], abs_tol= 5) and math.isclose(l1[1], l2[1], abs_tol= 5) and self.angle_difference_deg(l1[2], l2[2]) < math.radians(30):
            return True
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

    def command(self):
        print(self.vline_current)
        #if self.point_to_line_distance(320, 240, self.vline_current) > 30 or 
            
    def run(self):
        print(f"LineTracer start")
        threading.Thread(target=self._loop, daemon=True).start()


if __name__ == "__main__":
    cam = Camera()
    cam.run()
    lt = LineTracer(None, cam)
    lt.run()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("終了")