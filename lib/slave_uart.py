import serial
import threading
from threading import Lock
from struct import pack, unpack

class SlaveUART:
    def __init__(self, port, baudrate=921600, timeout=0.2):
        self.serial_port = serial.Serial(port, baudrate=baudrate, timeout=timeout)
        self.id = 2  # このスレーブのID
        self.bid = pack("B", self.id)
        self.stream_buffer = bytearray()
        self.mem_lock = Lock()
        self.memory = bytearray(256)
        self.memory[0] = 0
        self.memory[1] = False
        self.memory[2] = False
        self.memory[3:7] = bytearray(pack("<f", 0.0))
        self.memory[7:11] = bytearray(pack("<f", 0.0))
        self.memory[11] = 00 #pitch2
        self.memory[12] = 30 #pitch1
        self.memory[13] = 30 #yaw
        self.memory[14] = 30
        self.memory[15:19] = bytearray(pack("<f", 0.0))
        print(self.memory[13])

        self.memory_proto = {
            0x00: {"length": 1, "format": "<B"},
            0x01: {"length": 1, "format": "<?"},
            0x02: {"length": 1, "format": "<?"},
            0x03: {"length": 4, "format": "<f"},
            0x07: {"length": 4, "format": "<f"},
            0x0b: {"length": 1, "format": "<B"},
            0x0c: {"length": 1, "format": "<B"},
            0x0d: {"length": 1, "format": "<B"},
            0x0e: {"length": 1, "format": "<B"},
            0x0f: {"length": 4, "format": "<f"},
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
                if len(data) > 5:
                    self._parse_packet(data)

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
                            break
                        length = int(self.stream_buffer[i+3]) + 4
                        if i + length <= len(self.stream_buffer):
                            packet = self.stream_buffer[i:i+length]
                            packets.append(packet)
                            i += length
                        else:
                            break
                    else:
                        i += 1
                self.stream_buffer = self.stream_buffer[i:]
                for p in packets:
                    self._parse_packet(p)

    def _parse_packet(self, data):
        if len(data) < 6:
            return
        if data[0] != 0xFF or data[1] != 0xFF:
            return

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
            return

        match instruction:
            case 0x01:
                self._respond_status_packet(self.id)
                print("[INFO] PING received")

            case 0x02:
                if len(params) < 2:
                    return
                self._parse_read(params)

            case 0x03:
                if len(params) < 1:
                    return
                length = len(params) - 1
                self._parse_write(length, params)

            case 0x00:
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

    def send_status_packet(self, id):
        self.send_packet(id, 0x01, None)

    def _respond_status_packet(self, id):
        self.send_packet(id, 0x00, None)

    def run(self):
        print(f"SLAVE (ID={self.id}) start")
        threading.Thread(target=self._receive_loop_stream, daemon=True).start()