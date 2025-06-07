import serial
import threading
from threading import Lock
import queue
import time
from struct import pack, unpack
from picamera2 import Picamera2
import cv2
#from multiprocessing import Process, Lock
#from multiprocessing.sharedctypes import Value, Array

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

    def _parse_packet(self, data):
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
    
    def _loop(self):
        while True:
            with self.fc_lock:
                self.fc = self.picam.capture_array()
            with self.lc_lock:
                _, self.lc = self.cap.read()

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

# 普通に歩いていて

class LineTracer:
    def __init__(self, slave, camera):
        self.camera = camera
        self.enabled = True
        self.lsd = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD)
        
    
    def _loop(self):
        while True:
            img = self.camera.get_line_camera()
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray,(3,3),0)
            gray = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 5)

    def get_line_distance(self, l1, l2):
        p = (l2[0:2] + l2[2:4]) / 2
        ap = p - l1[0:2]
        ab = l1[2:4] - l1[0:2]
        ba = l1[0:2] - l1[2:4]
        bp = p - l1[2:4]
        ai_norm = np.dot(ap, ab)/norm(ab)
        neighbor_point = a + (ab)/norm(ab)*ai_norm
        return norm(p - neighbor_point)

    def get_line_dir_in_window(self, window):
        lines, width, prec, nfa = lsd.detect(window)
    
    def set_command_velocirty(self, x, yaw):
        self.slave.set_data(0x03, x)
        self.slave.set_data(0x07, yaw)
    
    def start_robot(self, start):
        self.slave.set_data(0x02, start)

    def walk_robot(self, start):
        self.slave.set_data(0x01, start)

    def get_window(self, src, center, size): #h, w
        return src[center[0]-size[0]//2:center[0]-size[0]//2+size[0],
                   center[1]-size[1]//2:center[1]-size[1]//2+size[1]]
    
    def set_enable(self, is_enabled):
        self.enabled = is_enabled

    def run(self):
        print(f"LineTracer start")
        threading.Thread(target=self._loop, daemon=True).start()


if __name__ == "__main__":
    slave = SlaveUART(port="/dev/ttyAMA0")  # 使用するシリアルポートを指定
    slave.run()
    cam = Camera()
    cam.run()
    #time.sleep(1)
    #slave.memory[2] = True
    try:
        while True:
            time.sleep(0.01)

    except KeyboardInterrupt:
        slave.serial_port.close()
        print("終了")