from pirc522 import RFID as rfid
import threading as thrd

class RFID:
    def __init__(self, cfg: dict, logger):
        self.logger = logger
        self.device = rfid(pin_irq=None)
        thrd.Thread(target=self).start()
        self.logger("RFID main thread started")
        self.uuid = [False, ""]
        self._flagIsPresented = 0

    def __call__(self):
        try:
            while 1:
                (error, data) = self.device.request()
                if not error:
                    if self._flagIsPresented: self._flagIsPresented = 5
                    (error, uuid) = self.device.anticoll()
                    if not self._flagIsPresented:
                        self._flagIsPresented = 5
                        self.uuid = [True, uuid]
                        self.logger(f"new card presented: {uuid}")
                else:
                    if self._flagIsPresented: self._flagIsPresented -= 1
        except Exception as e:
            self.logger(f"ERROR --- in RFID main thread: {e}")

    def presentedCard(self):
        val = self.uuid.copy()
        self.uuid[0] = False
        return val

if __name__ == "__main__":
    from pfand_types import Logger
    from time import sleep
    rc522 = RFID({}, Logger())
    while 1:
        sleep(1.5)
        print(rc522.presentedCard())