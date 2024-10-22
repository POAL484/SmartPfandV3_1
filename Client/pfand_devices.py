global is_emulator, HX711, RFID, createEmulator, AIR, Servo

is_emulator = False

def import_as(emulator=False):
    global is_emulator, HX711, RFID, createEmulator, AIR, Servo
    is_emulator = emulator
    if not emulator:
        from pfand_HX711 import HX711
        from pfand_rc522 import RFID
        from pfand_AIR import AIR
        from pfand_servo import Servo
    else:
        print("emulator")
        from pfand_emulator import HX711, RFID, createEmulator, AIR, Servo