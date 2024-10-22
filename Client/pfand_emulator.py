from flask import Flask, request

import threading as thrd

class ConfigVar:
    insts = {}
    def __init__(self, name, def_value):
        self.__class__.insts[name] = self
        self.__setattr__(name, def_value)
        self.name = name

class HX711:
    def __init__(self, cfg: dict, logger):
        self.weight = ConfigVar("weight", 0)

    def getWeight(self):
        try: return float(self.weight.weight)
        except ValueError: return 0
    
class RFID:
    def __init__(self, cfg: dict, logger):
        self.uuid = ConfigVar("uuid", "AAAAAAAA")
        self.isReaded = ConfigVar("read", False)

    def presentedCard(self):
        val = [self.isReaded.read, self.uuid.uuid]
        self.isReaded.read = False
        return val
    
class AIR:
    def __init__(self, cfg: dict, logger): pass
    def __call__(self): pass
    def close(self): pass

class Servo:
    def __init__(self, cfg: dict, logger): pass
    def __call__(self): pass
    def open(self): pass
    def close(self): pass

def createEmulator():
    app = Flask(__name__)

    @app.route("/")
    def updateSomething():
        for i in dict(request.args).keys():
            try: ConfigVar.insts[i].__setattr__(i, dict(request.args)[i])
            except KeyError: return "Does not found value with provided key"
            return f"Emulator update successfull, {i}: {dict(request.args)[i]}"
        
    thrd.Thread(target=app.run, args=("0.0.0.0", 5050)).start()