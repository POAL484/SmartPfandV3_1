import RPi.GPIO as GPIO
import time

class HX711:
    def __init__(self, cfg: dict, logger):
        self.logger = logger

        self.CLOCK_PIN = cfg['hx711']['clock']
        self.DATA_PIN = cfg['hx711']['data']

        GPIO.setmode(GPIO.BOARD)
        GPIO.setup(self.CLOCK_PIN, GPIO.OUT)
        GPIO.setup(self.DATA_PIN, GPIO.IN)

        self.offset = cfg['hx711']['offset']
        self.oneGramm = cfg['hx711']['1gramm']

    def available(self):
        return not GPIO.input(self.DATA_PIN)

    def readBit(self):
        GPIO.output(self.CLOCK_PIN, 1)
        GPIO.output(self.CLOCK_PIN, 0)
        return int(GPIO.input(self.DATA_PIN))
    
    def readByte(self):
        val = 0
        for i in range(8):
            val <<= 1
            val |= self.readBit()
        return val

    def readRawValue(self):
        while not self.available(): pass

        bytes = [self.readByte(), self.readByte(), self.readByte()]

        self.readBit()

        valNotSigned = ((bytes[0] << 16) | (bytes[1] << 8) | bytes[2])

        valSigned = -(valNotSigned & 0x80000) + (valNotSigned & 0x7fffff)

        return valSigned
    
    def readInGramms(self):
        val = self.readRawValue()
        val += self.offset
        val /= self.oneGramm
        return val

    def getWeight(self):
        times = 15

        weights = []
        for i in range(times): weights.append(self.readInGramms())
        weights.sort()
        return weights[(times-1)//2]

if __name__ == "__main__":
    from pfand_types import Logger
    hx = HX711({'hx711': {'clock': 6, 'data': 5, 'offset': -13750, '1gramm': 400}}, Logger())
    while 1:
        time.sleep(1)
        print(hx.getWeight(), hx.readRawValue())