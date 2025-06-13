import serial
import threading
from threading import Lock
import queue
import time
from struct import pack, unpack
import cv2
from stream import update_debug_frame, run_webserver

import numpy as np
import math
import uuid
import os
from lib.slave_uart import SlaveUART

from lib.obj_det import ObjectDetector
from lib.slave_uart import SlaveUART
from lib.camera import USBCamera, PiCamera

#速度： -50~50
#ターンスピード： 10～15 chous

framewidth = 320
frameheight = framewidth * 3 // 4  # 4:3のアスペクト比
currentmode = 0

# mode...0: 前進
# mode 1: ラインとしない
# mode...2: 後退

# zone...0: 

# TODO
#   色分けの反映
#   済 ボールへのにじり寄りを行った場合の退却
#   オブジェクトが見つからなかった場合の諦め
#   オブジェクトを探す場所の変更
#   スタート台への帰還

# register map
CURRENT_MODE = 0x00
WALK_ENABLE = 0x01
SERVO_ENABLE = 0x02
OBJ_SPEED = 0x03
TURN_OBJ_SPEED = 0x07
ARM_YAW_ANGLE = 0x0b
ARM_PITCH1_ANGLE = 0x0c
ARM_PITCH2_ANGLE = 0x0d
HAND_ANGLE = 0x0e
SUCTION_REF = 0x0f

class LineTracer:
    def __init__(self, slave, camera,
                 angle_threshold=3000, 
                 position_threshold=100, 
                 frame_check_count=20, 
                 debug_stream_enabled=False): # Added debug_stream_enabled
        
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

        self.tasks = ["linetrace", "count_vlines"] # "linetrace", "count_vlines"
        self.behavior = "goahead" #goahead, turn, stop
        self.mode = "forward" #forward, backward
        self.position = 0  
        # Position IDs
        # 0: start
        # 2: red_goal(line2)
        # 3: yellow_goal(line3)
        # 4: blue_goal(line4)
        # 5: object_zone(line5)

    def get_binary_image(self, debug=True):
        img = self.camera.get_line_camera()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray,(3,3),0)
        gray = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 10)
        debug_img = None
        if debug:
            debug_img = img.copy()
        return gray, debug_img
    
    def set_position(self, position):
        self.position = position

    async def goto(self, target_position):
        self.detected_hlines = []  # Reset detected horizontal lines
        self.detected_vlines = []  # Reset detected vertical lines
        self.hlines_crossed_count = 0  # Reset horizontal line crossed count
        self.hlines_current = {}

        if target_position == self.position:
            print(f"[DEBUG] Already at target position {target_position}. No action taken.")
            return
        elif target_position < self.position:
            print(f"[DEBUG] Moving backward to position {target_position}.")
            self.slave.set_data(CURRENT_MODE, 2)
            self.mode = "backward"
        else:
            print(f"[DEBUG] Moving forward to position {target_position}.")
            self.slave.set_data(CURRENT_MODE, 0)
            self.mode = "forward"
        
        while True:
            gray, debug_img = self.get_binary_image(debug=False)
            self.detect_vertical_line(gray, debug_img)
            self.detect_horizontal_line(gray, debug_img)
            self.update()
            if self.position == target_position:
                print(f"[DEBUG] Reached target position {target_position}.")
                break
            #self.command()
            if self.debug_stream_enabled and debug_img is not None: # Conditional call
                update_debug_frame(debug_img)
            await asyncio.sleep(0.02)  # Adjusted sleep time for asyncio

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
            gray = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 10)
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
        if self.slave.get_data(CURRENT_MODE) == (2,):
            centers = [
                (int(self.camera.width * 1.0 / 2), int(self.camera.height * 0.9)),
                (int(self.camera.width * 1.0 / 2), int(self.camera.height * 0.7)),
                (int(self.camera.width * 1.0 / 2), int(self.camera.height * 0.5))
            ]
        else:
            centers = [
                (int(self.camera.width * 1.0 / 2), int(self.camera.height * 0.1)),
                (int(self.camera.width * 1.0 / 2), int(self.camera.height * 0.3)),
                (int(self.camera.width * 1.0 / 2), int(self.camera.height * 0.5))
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
        currentmode = self.slave.get_data(CURRENT_MODE)
        if currentmode == (0,):
            centerheight = self.camera.height // 4
        elif currentmode == (2,) and self.hlines_crossed_count == 0:
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
        if self.mode == "backward":
            self.position -= 1
        elif self.mode == "forward":
            self.position += 1
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
            if self.slave.get_data(CURRENT_MODE) == (2,):
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
    
    def update(self):
        if self.vline_current is not None and len(self.vline_current) == 3:
            avg_angle_deg = np.degrees(self.vline_current[2])
            w = self.calculate_angular_velocity(avg_angle_deg)
        else:
            w = 0.0
        
        if not(w == 0):
            v = 0.0
        elif self.mode == "forward":
            v = 25.0
        elif self.mode == "backward":
            v = -25.0
        self.set_command_velocity(v, w)

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
        if ballcount == 0:
            LineToGo = 4
        else:
            LineToGo = 2
        if self.hlines_crossed_count <= LineToGo and self.slave.get_data(CURRENT_MODE) == (0,):
            if self.hlines_crossed_count <= 1:
                max_linear_speed = 20
            elif self.hlines_crossed_count > 3:
                max_linear_speed = 30
            else:
                max_linear_speed = 40
            if not(w == 0):
                v = 0.0
            else:
                v = max_linear_speed
        elif self.hlines_crossed_count <= 2 and self.slave.get_data(CURRENT_MODE) == (2,):
            max_linear_speed = -20
            if not(w == 0):
                v = 0.0
            else:
                v = max_linear_speed
        else:
            v = 0  # Stop moving forward if horizontal line count exceeds 5
            self.slave.set_data(WALK_ENABLE, False)
            self.set_command_velocity(0, 0)
            sleep_time = 0.5
            print(f"[DEBUG] Horizontal line count exceeded, stopping forward movement. Waiting for {sleep_time:.2f} seconds.")
            time.sleep(sleep_time)
            self.slave.set_data(WALK_ENABLE, False)
            self.slave.set_data(CURRENT_MODE, 1)
            currentmode = (1,)
            while currentmode == (1,):
                time.sleep(1)
                currentmode = self.slave.get_data(CURRENT_MODE)
            self.hlines_crossed_count = 0

        #print(f"[DEBUG] Linear velocity (v): {v:.2f}, Angular velocity (w): {w:.2f}")
        self.set_command_velocity(v, w)

        #if self.point_to_line_distance(320, 240, self.vline_current) > 30 or 
            

    def run(self):
        print(f"LineTracer start")
        threading.Thread(target=self._loop, daemon=True).start()


import asyncio
import time

class DecisionMaker:
    def __init__(self, slave, objdet, cam, picam, linetrace, time_limit=300): #予選:300秒, 決勝:600秒
        self.slave = slave
        self.objdet = objdet
        self.cam = cam
        self.picam = picam
        self.lt = linetrace
        self.time_limit = time_limit
        self.start_time = time.time()

        self.ballcount = 0
        self.search_pos = [-10, 0, 0] #advance length, turn angle, advance length
    
    def run(self):
        print("DecisionMaker start")
        asyncio.run(self._main())
        
    async def _main(self):
        await self.launch()
        await self.goto_object_zone()
        while True:
            object = await self.search_object()
            match object[0]: #class id
                case 0: # red ball
                    print("Found red ball")
                    await self.catch_ball_or_can(object[1:])
                case 1: # blue ball
                    print("Found blue ball")
                    await self.catch_ball_or_can(object[1:])
                case 3: # yellow can
                    print("Found can")
                    await self.catch_ball_or_can(object[1:])
                case _:
                    print("No object found, searching again")
            self.bring_object_to_goal(object[0])
            await self.goto_object_zone()
    
    async def launch(self):
        print("launch phase: 自由ボール捨てる")
        await asyncio.sleep(0.2)
        self.slave.set_data(SERVO_ENABLE, True) #servo on
        await asyncio.sleep(3)
        self.slave.set_data(SUCTION_REF, 8.0)
        self.slave.set_data(ARM_PITCH2_ANGLE, 145)
        #アーム展開
        self.slave.set_data(ARM_YAW_ANGLE, 40)
        await asyncio.sleep(0.5)
        self.slave.set_data(ARM_PITCH1_ANGLE, 90)
        await asyncio.sleep(1)
        # slave.set_data(SUCTION_REF, 0.0)
        self.slave.set_data(ARM_PITCH1_ANGLE, 180)
        await asyncio.sleep(0.3)
        self.slave.set_data(ARM_YAW_ANGLE, 100)
        await asyncio.sleep(1)
        #slave.set_data(ARM_YAW_ANGLE, 105)
        self.slave.set_data(SUCTION_REF, 0.0)
        #アーム収納
        self.slave.set_data(ARM_PITCH1_ANGLE, 90)
        await asyncio.sleep(1)
        self.slave.set_data(ARM_YAW_ANGLE, 40)
        await asyncio.sleep(0.5)
        self.slave.set_data(ARM_PITCH1_ANGLE, 0)
        await asyncio.sleep(0.3)
        self.slave.set_data(ARM_PITCH2_ANGLE, 105)
        self.slave.set_data(ARM_YAW_ANGLE, 0)
    
    async def search_object(self):
        init_pos = self.search_pos[0]
        await self.walk(init_pos)
        while True:
            self.search_pos[1] -= 30
            await self.turn(-30)
            obj_data = await self.recognize_object()
            if obj_data is not None:
                return obj_data
            
            self.search_pos[1] += 60
            await self.turn(60)
            obj_data = await self.recognize_object()
            if obj_data is not None:
                return obj_data
            
            self.search_pos[1] -= 30
            await self.turn(-30)

            self.search_pos[0] += 20
            await self.walk(20)
        
    async def recognize_object(self):
        print("recognize_object phase: オブジェクト認識")
        img = picam.get_front_camera()
        if img is None:
            print("cannot aquire camera image!")
            return None
        
        outputs = objdet.predict(img)
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
                bottom_center = [(bbox[0] + bbox[2]) / 2, bbox[3]]
                #print(f"最大面積オブジェクト（クラス{label}）: 2D中心={center}, 面積={area}")
                theta, _ = picam.fc_convert_2dpos_to_3d(center)
                _, position = picam.fc_convert_2dpos_to_3d(bottom_center)
                if np.linalg.norm(position) < 0.4:
                    print(f"オブジェクト認識成功: クラス={label}, 角度={theta:.2f}, 位置={position}")
                    return label, theta, position
        return None

    async def goto_object_zone(self):
        print("goto_object_zone phase: オブジェクトゾーンへ移動")
        await self.lt.goto(5)
    
    async def bring_object_to_goal(self, class_id):
        print(f"bring_object_to_goal phase: オブジェクトをゴールへ持っていく (クラスID: {class_id})")
        await self.walk(-self.search_pos[2])
        await self.turn(-self.search_pos[1])
        await self.walk(10 - self.search_pos[0])
        self.search_pos[1] = 0; self.search_pos[2] = 0

        #start linetrace
        match class_id: #class id
            case 0: # red ball
                await self.lt.goto(2)
                await self.release_ball_or_can()
            case 1: # blue ball
                await self.lt.goto(4)
                await self.release_ball_or_can()
            case 3: # yellow can
                await self.lt.goto(3)
                await self.release_ball_or_can()
            case _:
                # other object detected.
                pass
        
    async def walk(self, length, vel=15):
        if length == 0:
            return 0
        elif length > 0:
            vel = abs(vel)
        else:
            vel = -abs(vel)
        self.slave.set_data(WALK_ENABLE, True)
        self.slave.set_data(OBJ_SPEED, vel)
        self.slave.set_data(TURN_OBJ_SPEED, 0)
        await asyncio.sleep(length / vel)
        self.slave.set_data(OBJ_SPEED, 0)
        self.slave.set_data(WALK_ENABLE, False)
        await asyncio.sleep(0.5)
        return length

    async def turn(self, angle, omega=15):
        if angle == 0:
            return 0
        elif angle > 0:
            omega = abs(omega)
        if angle < 0:
            omega = -abs(omega)
        self.slave.set_data(WALK_ENABLE, True)
        self.slave.set_data(OBJ_SPEED, 0)
        self.slave.set_data(TURN_OBJ_SPEED, omega)
        await asyncio.sleep(angle / omega)
        self.slave.set_data(TURN_OBJ_SPEED, 0)
        self.slave.set_data(WALK_ENABLE, False)
        await asyncio.sleep(0.5)
        return angle
    
    async def catch_ball_or_can(self, data):
        print("catch_ball_or_can phase: ボール・缶をキャッチ")
        theta, position = data
        #近くに寄るプログラム
        self.slave.set_data(ARM_PITCH2_ANGLE, int(np.clip(-theta*2/4+105,0,180)))
        #アーム展開
        self.slave.set_data(ARM_YAW_ANGLE, 40)
        await asyncio.sleep(0.5)
        self.slave.set_data(ARM_PITCH1_ANGLE, 90)
        await asyncio.sleep(1)
        self.slave.set_data(ARM_PITCH1_ANGLE, 180)
        await asyncio.sleep(0.3)
        self.slave.set_data(ARM_YAW_ANGLE, 80)
        await asyncio.sleep(1)
        self.slave.set_data(SUCTION_REF, 0.99)
        await asyncio.sleep(1)
        self.slave.set_data(ARM_YAW_ANGLE, 100)
        await asyncio.sleep(0.5)
        self.slave.set_data(ARM_YAW_ANGLE, 90)
        await asyncio.sleep(0.5)
        #アーム収納
        self.slave.set_data(ARM_PITCH1_ANGLE, 90)
        await asyncio.sleep(1)
        self.slave.set_data(ARM_YAW_ANGLE, 40)
        await asyncio.sleep(0.5)
        self.slave.set_data(ARM_PITCH1_ANGLE, 0)
        await asyncio.sleep(0.3)
        self.slave.set_data(ARM_PITCH2_ANGLE, 105)
        self.slave.set_data(ARM_YAW_ANGLE, 0)
        await asyncio.sleep(1.5)
        self.slave.set_data(SUCTION_REF, 0.7)
    
    async def release_ball_or_can(self, label):
        self.slave.set_data(WALK_ENABLE, False)
        self.slave.set_data(SUCTION_REF, 8.0)
        self.slave.set_data(ARM_PITCH2_ANGLE, 195)
        #アーム展開
        self.slave.set_data(ARM_YAW_ANGLE, 40)
        await asyncio(0.5)
        self.slave.set_data(ARM_PITCH1_ANGLE, 90)
        await asyncio(1)
        self.slave.set_data(SUCTION_REF, 0.0)
        self.slave.set_data(ARM_PITCH1_ANGLE, 180)
        await asyncio(0.3)
        self.slave.set_data(ARM_YAW_ANGLE, 100)
        await asyncio(1)
        #slave.set_data(ARM_YAW_ANGLE, 105)
        self.slave.set_data(SUCTION_REF, 0.0)
        #アーム収納
        self.slave.set_data(ARM_PITCH1_ANGLE, 90)
        await asyncio(1)
        self.slave.set_data(ARM_YAW_ANGLE, 40)
        await asyncio(0.5)
        self.slave.set_data(ARM_PITCH1_ANGLE, 0)
        await asyncio(0.3)
        self.slave.set_data(ARM_PITCH2_ANGLE, 105)
        self.slave.set_data(ARM_YAW_ANGLE, 0)


if __name__ == "__main__":
    #threading.Thread(target=run_webserver, daemon=True).start()
    slave = SlaveUART(port="/dev/ttyAMA0")  # 使用するシリアルポートを指定
    slave.run()
    cam = USBCamera(width=320, height=240)
    cam.run()
    picam = PiCamera()
    picam.run()
    objdet = ObjectDetector("/home/teba/Programs/inrof2025/python/lib/masters.onnx")
    lt = LineTracer(slave, cam, debug_stream_enabled=False)
    dm = DecisionMaker(slave, objdet, cam, picam, lt, time_limit=300)
    dm.run()