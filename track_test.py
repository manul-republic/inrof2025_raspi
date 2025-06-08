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

#速度： -50~50
#ターンスピード： 10～15 chous

class SlaveUART:
    def __init__(self, port, baudrate=921600, timeout=0.2):
        self.serial_port = serial.Serial(port, baudrate=baudrate, timeout=timeout)
        self.id = 2  # このスレーブのID
        self.bid = pack("B", self.id)
        #ff 03 in little-endian (<)
        self.stream_buffer = bytearray()
        self.mem_lock = Lock()
        self.memory = bytearray(256)
        self.memory[0] = 0 #pack("B", 5)
        self.memory[1] = False #1で歩行モードに移動　0で停止
        self.memory[2] = False #1で励磁開始
        self.memory[3:7] = bytearray(pack("<f", 0.0))
        self.memory[7:11] = bytearray(pack("<f", 0.0)) #pack("B", 8)

        self.memory_proto = {
            0x00: {"length": 1, "format":"<B"},     #現在の制御モード uint8_t 1バイト
            0x01: {"length": 1, "format":"<?"},     #1で歩行モードに移動　0で停止
            0x02: {"length": 1, "format":"<?"},     #1で励磁開始
            0x03: {"length": 4, "format":"<f"},     #x方向速度 -50~50
            0x07: {"length": 4, "format":"<f"},     #yaw軸各速度 10～15が望ましい 低すぎると動かないらしい
        }

    def _checksum(self, data):
        return (~sum(data)) & 0xFF
    
    def get_data(self, key):
        with self.mem_lock:
            length = self.memory_proto[key]["length"]
            value = unpack(self.memory_proto[key]["format"], self.memory[key:key+length]) 
        return value
    
    def set_data(self, key, value):
        with self.mem_lock:
            length = self.memory_proto[key]["length"]
            self.memory[key:key+length] = pack(self.memory_proto[key]["format"], value)

    def _receive_loop(self):
        while True:
            if self.serial_port.in_waiting > 0:
                data = self.serial_port.read(self.serial_port.in_waiting)
                #print(f"[DEBUG] received: f{bytes(data).hex()}")
                if len(data) > 5:
                    self._parse_packet(data)
            #time.sleep(0.002)
    
    #いっぺんにデータが来ても大丈夫バージョン 適切にパケット分けしたり
    def _receive_loop_stream(self):
        while True:
            if self.serial_port.in_waiting > 0:
                new_data = self.serial_port.read(self.serial_port.in_waiting)
                self.stream_buffer += new_data
                packets = []
                i = 0
                while i + 3 < len(self.stream_buffer):
                    if self.stream_buffer[i] == 0xFF and self.stream_buffer[i+1] == 0xFF:
                        if i + 4 >= len(self.stream_buffer):
                            break  # データ不足、次の受信で補完
                        length = int(self.stream_buffer[i+3]) + 4
                        if i + length <= len(self.stream_buffer):
                            packet = self.stream_buffer[i:i+length]
                            packets.append(packet)
                            i += length  # 次のパケットへ
                        else:
                            break  # データ不足、次の受信で補完
                    else:
                        i += 1  # ヘッダを探し進む
                self.stream_buffer = self.stream_buffer[i:]
                for p in packets:
                    #print(f"[DEBUG] received: f{bytes(p).hex()}")
                    self._parse_packet(p)
            #time.sleep(0.002)

    def _parse_write(self, params):
        with self.mem_lock:
            addr = int(params[0])
            self.memory[addr:addr + len(params[1:])] = bytearray(params[1:])
        self.send_packet(self.id, 0x00, None)
        if len(data) < 6:
            return  # 不完全パケット無視
        if data[0] != 0xFF or data[1] != 0xFF:
            return  # ヘッダー不正

        recv_id = data[2]
        length = data[3]
        instruction = data[4]
        params = data[5:-1]
        checksum = data[-1]
        calc_checksum = self._checksum(data[2:-1])
        if checksum != calc_checksum:
            print(f"[ERROR] checksum is irrelevant. checksum: {checksum}, calculated: {calc_checksum}")
            return

        if recv_id != self.id and recv_id != 0xFE:
            return  # ID不一致

        match instruction:
            case 0x01: # PING
                self._respond_status_packet(self.id) 
                print("[INFO] PING received")

            case 0x02:  # READ DATA
                if len(params) < 2:
                    return
                self._parse_read(params)

            case 0x03:  # WRITE DATA
                if len(params) < 1:
                    return
                self._parse_write(params)
            
            case 0x00:  #response or something, スレーブには通常来ない はず？
                print(f"[DEBUG] 0x00 received: f{bytes(data).hex()}")

    def _parse_read(self, params):
        with self.mem_lock:
            addr = int(params[0])
            length = int(params[1])
            ps = self.memory[addr:addr+length]
        self.send_packet(self.id, 0x00, ps)
    
    def _parse_write(self, length, params):
        with self.mem_lock:
            addr = int(params[0])
            self.memory[addr:addr+length] = bytearray(params[1:])
            #print(f"[DEBUG] WRITE cmd received: addr: {addr}, data: {params[1:]}")
        self.send_packet(self.id, 0x00, None)

    def send_packet(self, id, inst, params):
        packet = bytes([0xFF, 0xFF, id])
        if params is None:
            packet += pack("B", 2)
        else:
            packet += pack("B", len(params)+2)
        packet += pack("B", inst)
        if params is not None:
            packet += params
        checksum = self._checksum(packet[2:])
        packet += pack("B", checksum)
        self.serial_port.write(bytes(packet))
        #print(f"[SEND] STATUS: {bytes(packet).hex()}")

    def send_status_packet(self, id):
        self.send_packet(id, 0x01, None)
    
    def _respond_status_packet(self, id):
        self.send_packet(id, 0x00, None)

    def run(self):
        print(f"SLAVE (ID={self.id}) start")
        #threading.Thread(target=self._receive_loop, daemon=True).start()
        threading.Thread(target=self._receive_loop_stream, daemon=True).start()

class Camera:
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
            a = self.lc.copy()
        return a
    
    def get_front_camera(self):
        with self.fc_lock:
            a = self.fc.copy()
        return a
    
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
    cam = Camera()
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