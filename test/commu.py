import serial
import threading
from threading import Lock
import queue
import time
from struct import pack, unpack
#from multiprocessing import Process, Lock
#from multiprocessing.sharedctypes import Value, Array

#dictでデータ管理してるけどメモリマップの方が普通にいいかもしれない
#ただどうせformat情報が必要になるしこれでもいいんじゃね

class SlaveUART:
    def __init__(self, port, queue, baudrate=115200, timeout=0.1):
        self.serial_port = serial.Serial(port, baudrate=baudrate, timeout=timeout)
        self.lock = Lock()
        self.id = 2  # このスレーブのID
        self.bid = pack("B", self.id)
        #self.memory = [0] * 256  # 256バイトのメモリテーブル
        #self.memory[0x05] = self.id  # IDをメモリにセット（例）
        #03 ff in big-endian (>)
        #ff 03 in little-endian (<)
        # lockいる？常にこのスレッドしか使わないならデータ競合はしないよね
        # 代わりにmemoryを守る必要がある
        self.stream_buffer = bytearray()
        self.memory = {
            0x01: {"data": None, "length": 1, "format":"<B"},
            0x02: {"data": None, "length": 2, "format":"<h"},
            0x03: {"data": None, "length": 1, "format":"<B"},
        }

    def _checksum(self, data):
        return (~sum(data)) & 0xFF

    def _receive_loop(self):
        while True:
            if self.serial_port.in_waiting > 0:
                with self.lock:
                    data = self.serial_port.read(self.serial_port.in_waiting)
                self._parse_packet(data)
                #print(f"recv packet: {data}")
            time.sleep(0.002)
    
    #いっぺんにデータが来ても大丈夫バージョン 適切にパケット分けしたり
    def _receive_loop_stream(self):
        while True:
            if self.serial_port.in_waiting > 0:
                with self.lock:
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
                    self._parse_packet(p)
            time.sleep(0.002)


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
                if params[0] != 0x00:
                    self._respond_status_packet(params[0])
                    print("[INFO] PING received")
                else:
                    print("[INFO] PING returned")

            case 0x02:  # READ DATA
                if len(params) < 2:
                    return
                addrs = self._parse_read(params)
                self._respond_read_data(addrs)

            case 0x03:  # WRITE DATA
                if len(params) < 1:
                    return
                self._parse_write(params)
                #self._respond_status_packet(recv_id)
                """addr = params[0]
                data_bytes = params[1:]
                print(f"[INFO] WRITE DATA received: addr={addr}, data={data_bytes}")
                for i, b in enumerate(data_bytes):
                    self.memory[addr + i] = b"""
            
            case 0x00:  #response or something
                print(f"[DEBUG] 0x00 received: f{bytes(data).hex()}")

    
    def _parse_read(self, params):
        addrs = []
        for i in range(len(params) // 2):
            addr = int(params[i*2])
            addrs.append(addr)
            if self.memory[addr]["length"] != int(params[i*2+1]):
                raise ValueError("[ERROR] Illegal command with different data length received.")
            print(f"[DEBUG] READ cmd received: addr: {addr}")
        return addrs
    
    def _parse_write(self, params):
        counter = 0
        while counter < len(params):
            addr = int(params[counter])
            counter += 1
            length = self.memory[addr]["length"]
            data = unpack(self.memory[addr]["format"], params[counter:counter+length])[0] #return tuple
            self.memory[addr]["data"] = data
            print(f"[DEBUG] WRITE cmd received: addr: {addr}, data: {data}")
            counter += length

    def send_packet(self, id, inst, params):
        packet = bytes([0xFF, 0xFF, id])
        packet += pack("B", len(params)+2)
        packet += pack("B", inst)
        packet += params
        checksum = self._checksum(packet[2:])
        packet += pack("B", checksum)
        with self.lock:
            self.serial_port.write(bytes(packet))
        print(f"[SEND] STATUS: {bytes(packet).hex()}")

    def send_status_packet(self, id):
        self.send_packet(id, 0x01, self.bid)
    
    def _respond_status_packet(self, id):
        self.send_packet(id, 0x01, bytes([0x00]))

    def _respond_read_data(self, addrs):
        packet = None
        for addr in addrs:
            data = pack(self.memory[addr]["format"], self.memory[addr]["data"]) #return tuple
            if not packet: packet = data
            else: packet += data
        self.send_packet(self.id, 0x00, packet)
        """data = self.memory[addr:addr+length]
        packet = [0xFF, 0xFF, self.id, length+2, 0x00] + data
        checksum = self._checksum(packet[2:])
        packet += (checksum)
        self.send_packet(self.id, 0x00, packet)"""

    def run(self):
        print(f"SLAVE (ID={self.id}) start")
        #threading.Thread(target=self._receive_loop, daemon=True).start()
        threading.Thread(target=self._receive_loop_stream, daemon=True).start()


if __name__ == "__main__":
    slave = SlaveUART(port="/dev/ttyAMA0", queue=None)  # 使用するシリアルポートを指定
    slave.run()
    time.sleep(0.1)
    print()
    #slave.send_status_packet(0x02)
    #slave.send_packet(0x02, 0x03, bytes([0x01, 0x22]))
    #time.sleep(0.01)
    #slave.send_packet(0x02, 0x03, bytes([0x02, 0x22, 0x22]))
    #slave.send_packet(0x02, 0x03, bytes([0x01, 0x22]))
    slave.send_packet(0x02, 0x03, bytes([0x01, 0x22]))
    #time.sleep(0.01)
    slave.send_packet(0x02, 0x02, bytes([0x01, 0x01]))
    #time.sleep(0.01)
    slave.send_packet(0x02, 0x03, bytes([0x02, 0x22, 0x22]))
    #time.sleep(0.01)
    slave.send_packet(0x02, 0x02, bytes([0x01, 0x01, 0x02, 0x02]))

    try:
        while True:
            time.sleep(1)
            slave.send_status_packet(0x02)

    except KeyboardInterrupt:
        slave.serial_port.close()
        print("終了")