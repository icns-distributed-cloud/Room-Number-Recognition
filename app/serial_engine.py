import serial

class SerialEngine:
    def __init__(self, port, baudrate):
        self.port = port
        self.baudrate = baudrate

        self.connect()

    def connect(self):
        self.serial = serial.Serial(port=self.port, baudrate=self.baudrate)

    def write(self, msg):
        self.serial.write(msg)

    def close(self):
        self.serial.close()