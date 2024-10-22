import RPi.GPIO as GPIO
import threading as thrd
import time

class AIR:
    def __init__(self, cfg: dict, logger):
        self.logger = logger

        self.pin = cfg['air']['pin']
        self.air_time = cfg['air']['time']

        GPIO.setmode(GPIO.BOARD)
        GPIO.setup(self.pin, GPIO.OUT)

        self.logger("air inited")

    def __call__(self):
        GPIO.output(self.pin, 1)
        self.logger("Air started")
        thrd.Thread(target=self.close).start()

    def close(self):
        time.sleep(self.air_time)
        GPIO.output(self.pin, 0)
        self.logger("Air closed")