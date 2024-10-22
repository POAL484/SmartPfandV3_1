import json
import pygame as pg
import math
import threading as thrd

from pfand_types import *
import pfand_colors as colors
from pfand_ws import WsClient, WsState
from pfand_neural import *

import pfand_devices as dvs
dvs.import_as(emulator=False)

pg.init()

class Screen:
    def __init__(self, app):
        self.app = app
        self.root = app.root

    def toScreen(self, screenClass):
        self.app.screen = screenClass(self.app)
        self.app.logger(f"App changed screen to {str(screenClass.__name__)}")

    def call(self):
        #print(self.app.wsclient.msg)
        if not self.app.neural.ws_send_send is None:
            self.app.wsclient.neural_call(self.app.neural.ws_send_send)
            self.app.neural.ws_send_send = None
        pred = self.app.wsclient.find("neural.prediction.finish")
        self.app.wsclient.find("neural.prediction.pcg") 
        self.app.wsclient.find("neural.prediction.start")
        self.app.wsclient.find("ping")
        if not pred is None:
            self.app.neural.ws_send_recv = pred['data']

class InitScreen(Screen):   
    def __call__(self):
        self.root.fill((0, 0, 0))
        es = EventState()
        #self.call()
        Text(self.root, es, 100, 100, "Welcome to pfand graphics v3", 25, (255, 255, 255), 'Arial', Anchor.LEFT)
        Text(self.root, es, 100, 150, f"Version: first build", 25, (255, 255, 255), 'Arial', Anchor.LEFT)
        Text(self.root, es, 100, 200, f"Machine id: {self.app.config['machine_id']}", 25, (255, 255, 255), 'Arial', Anchor.LEFT)
        Text(self.root, es, 100, 250, f"Continue in {round(15-(time.time()-app.delta_time))}", 25, (255, 255, 255), 'Arial', Anchor.LEFT)
        Text(self.root, es, 750, 100, f"Ws State: {self.app.wsclient.state.name}", 25, (255, 255, 255), 'Arial', Anchor.LEFT)
        Text(self.root, es, 750, 150, f"Storage mode: {self.app.storageMode.name}", 25, (255, 255, 255), 'Arial', Anchor.LEFT)
        Text(self.root, es, 750, 200, f"HX711 value: {self.app.hx711.getWeight()}", 25, (255, 255, 255), 'Arial', Anchor.LEFT)
        Text(self.root, es, 750, 250, f"RFID value: {self.app.rfid.presentedCard()}", 25, (255, 255, 255), 'Arial', Anchor.LEFT)
        Text(self.root, es, 750, 300, f"Neural state: {self.app.neural.state.name}", 25, (255, 255, 255), 'Arial', Anchor.LEFT)
        lastlogs = self.app.logger.logs[::-1]
        for i in range(15):
            if i > len(lastlogs)-1: break
            Text(self.root, es, 120, 300+(i*30), lastlogs[i], 20, (240, 240, 240), 'Arial', Anchor.LEFT)
        if     (self.app.wsclient.state != WsState.READY \
            and self.app.wsclient.state != WsState.MESSAGE \
            and self.app.wsclient.state != WsState.FAILED_AUTHDATA) \
            or (self.app.neural.state != NeuralState.INITED \
            and self.app.neural.state != NeuralState.FAILED_DOUBLE): app.delta_time = time.time()
        else:
            #self.app.logger("timer started")
            match self.app.wsclient.state:
                case WsState.READY | WsState.MESSAGE:
                    self.app.storageMode = StorageMode.ONLINE
                case WsState.FAILED_AUTHDATA:
                    self.app.storageMode = StorageMode.OFFLINE
        if self.app.neural.state == NeuralState.NOT_INITED or self.app.neural.state == NeuralState.FAILED: thrd.Thread(target=self.app.neural.init).start()
        if time.time() - app.delta_time > 3: self.toScreen(IdleScreen)

class IdleScreen(Screen):
    def __call__(self):
        self.root.fill((255, 255, 255))
        es = EventState() # чем гуще лес шкибиди доп ес ес
        self.call()

        SinGraf(self.root, es, 20, 350, colors.WATER_100, 0.022, 15)
        SinGraf(self.root, es, 20, 275, colors.WATER_200, 0.0175, 30)
        SinGraf(self.root, es, 20, 200, colors.WATER_300, 0.013, 50)

        #anim_delt_norm = self.app.animation.copy()
        #anim_delt_norm['delta'] -= 1707300952.370399

        #Text(self.root, es, 300, 300, str(anim_delt_norm), 16, (0, 0, 0), 'Arial', Anchor.LEFT)
        #Text(self.root, es, 300, 340, str(time.time()-1707300952.370399), 16, (0, 0, 0), 'Arial', Anchor.LEFT)

        Button(self.root, es, self.app.width-150, 150, 250, 250, lambda: None, (255, 255, 255), Anchor.RIGHT)
        pg.draw.circle(self.root, (60, 60, 60), (self.app.width-125, 125), 225/3, 15)
        pg.draw.rect(self.root, (60, 60, 60), pg.Rect(self.app.width-125-7.5, 125-10, 15, 55))
        pg.draw.rect(self.root, (60, 60, 60), pg.Rect(self.app.width-125-7.5, 125-45, 15, 15))

        Text(self.root, es, self.app.width//2, 125, "Помести банку в банкоприемник", 64, (35, 35, 35), 'Arial', Anchor.CENTER, True)

        #Bank(self.root, es, self.app.width//2, self.app.height//2, 0.6, 0, 0)

        '''weight = self.app.hx711.getWeight()
        if weight > self.app.config['min_bank_weight'] and weight < self.app.config['max_bank_weight']:
            if self.app.animation['name'] == None:
                self.app.animation = {"name": "bank_started_weight", "delta": time.time(), "anim_time": 1.4}
            if self.app.animation['name'] == "bank_ended_weight":
                self.app.animation = {"name": "bank_started_weight", "delta": time.time() - (1.4 - (time.time() - self.app.animation['delta'])), "anim_time": 1.4}
            if self.app.animation['name'] == "bank_started_weight" and time.time() - self.app.animation['delta'] >= self.app.animation['anim_time']:
                self.app.animation = {"name": "bank_right_weight", "delta": time.time(), "anim_time": 1}                
            if self.app.animation['name'] == "bank_started_weight":
                Bank(self.root, es, -75 + math.sin((time.time() - self.app.animation['delta'])*(math.pi/2/self.app.animation['anim_time']))*250, self.app.height//2-100, 1.6 + math.sin((time.time() - self.app.animation['delta'])*(math.pi/2/self.app.animation['anim_time']))*1., 0, 0)
            elif self.app.animation['name'] == "bank_right_weight":
                Bank(self.root, es, 175, self.app.height//2-100, 2.6, 0, 0)
        elif weight > self.app.config['max_bank_weight']:
            if self.app.animation['name'] == None:
                self.app.animation = {"name": "bank_overweight_started_weight", "delta": time.time(), "anim_time": 1.4}
            ##if self.app.animation['name'] == "bank_right_weight"
            Bank(self.root, es, -75 + math.sin((time.time() - self.app.animation['delta'])*(math.pi/2/self.app.animation['anim_time']))*250, self.app.height//2-100, 1.6 + math.sin((time.time() - self.app.animation['delta'])*(math.pi/2/self.app.animation['anim_time'])), math.sin((time.time() - self.app.animation['delta'])*(math.pi/2/self.app.animation['anim_time']))*-75, 0)
        elif weight < self.app.config['min_bank_weight'] and self.app.animation['name'] != None:
            if self.app.animation['name'] == "bank_started_weight":
                self.app.animation = {"name": "bank_ended_weight", "delta": time.time() - (1.4 - (time.time() - self.app.animation['delta'])), "anim_time": 1.4}
            if self.app.animation['name'] == "bank_right_weight":
                self.app.animation = {"name": "bank_ended_weight", "delta": time.time(), "anim_time": 1.4}
            if self.app.animation['name'] == "bank_ended_weight" and time.time() - self.app.animation['delta'] >= self.app.animation['anim_time']:
                self.app.animation = {"name": None, "delta": time.time(), "anim_time": 1}
            else: Bank(self.root, es, -75 + math.sin(math.pi/2 - (time.time() - self.app.animation['delta'])*(math.pi/2/self.app.animation['anim_time']))*250, self.app.height//2-100, 1.6 + math.sin(math.pi/2 - (time.time() - self.app.animation['delta'])*(math.pi/2/self.app.animation['anim_time']))*1., 0, 0)           '''
        weight = self.app.hx711.getWeight()
        self.app.bankAnimation(weight, self.root, es)
        if weight > self.app.config['min_bank_weight'] and weight < self.app.config['max_bank_weight']:
            Text(self.root, es, self.app.width//2, self.app.height//2-100, "Сейчас ИИ определит твою банку...", 44, (0, 0, 0), 'Arial', Anchor.CENTER)
            if self.app.bankWorkState == BankWorkState.NOTHING: self.app.bankWorkState = BankWorkState.NEED_START_NEURAL
        if self.app.bankWorkState == BankWorkState.CARD: self.toScreen(CardScreen)
        if self.app.bankWorkState == BankWorkState.NEURAL_FAIL:
            Text(self.root, es, self.app.width//2, self.app.height//2+100, "ИИ не определил банку", 44, (0, 0, 0), 'Arial', Anchor.CENTER)
        if weight < self.app.config['min_bank_weight'] or weight > self.app.config['max_bank_weight']: self.app.bankWorkState = BankWorkState.NOTHING

class CloseIdleAnimationScreen(Screen):
    def __init__(self, app):
        super().__init__(app)
        self.animDelt = time.time()
        self.animDuration = 1.4

    def __call__(self):
        self.root.fill((255, 255, 255))
        es = EventState() # чем гуще лес шкибиди доп ес ес
        self.call()

        SinGraf(self.root, es, 20, 350 - (((time.time() - self.animDelt)/self.animDuration))*450, colors.WATER_100, 0.022, 15)
        SinGraf(self.root, es, 20, 275 - (((time.time() - self.animDelt)/self.animDuration))*375, colors.WATER_200, 0.0175, 30)
        SinGraf(self.root, es, 20, 200 - (((time.time() - self.animDelt)/self.animDuration))*300, colors.WATER_300, 0.013, 50)

        Button(self.root, es, self.app.width-150, 150 - ((time.time() - self.animDelt)/self.animDuration)*250, 250, 250, lambda: None, (255, 255, 255), Anchor.RIGHT)
        pg.draw.circle(self.root, (60, 60, 60), (self.app.width-125, 125 - ((time.time() - self.animDelt)/self.animDuration)*225), 225/3, 15)
        pg.draw.rect(self.root, (60, 60, 60), pg.Rect(self.app.width-125-7.5, 125-10 - ((time.time() - self.animDelt)/self.animDuration)*235, 15, 55))
        pg.draw.rect(self.root, (60, 60, 60), pg.Rect(self.app.width-125-7.5, 125-45 - ((time.time() - self.animDelt)/self.animDuration)*270, 15, 15))

        Text(self.root, es, self.app.width//2, 125 - ((time.time() - self.animDelt)/self.animDuration)*225, "Помести банку в банкоприемник", 64, (35, 35, 35), 'Arial', Anchor.CENTER, True)
        
        Text(self.root, es, self.app.width//2, self.app.height//2-100 - ((time.time() - self.animDelt)/self.animDuration)*(self.app.height//2), "Банка ура оооооо да банка ахереть это че банка", 44, (0, 0, 0), 'Arial', Anchor.CENTER)

        if (time.time() - self.animDelt) >= self.animDuration: self.toScreen(OpenCardAnimationScreen)
        
class OpenCardAnimationScreen(Screen):
    def __init__(self, app):
        super().__init__(app)
        self.animDelt = time.time()
        self.animDuration = 1.4

    def __call__(self):
        self.root.fill((255, 255, 255))
        es = EventState() # чем гуще лес шкибиди доп ес ес
        self.call()

        #SinGraf(self.root, es, 20, 350 - (((time.time() - self.animDelt)/self.animDuration))*450, colors.WATER_100, 0.022, 15)
        #SinGraf(self.root, es, 20, 275 - (((time.time() - self.animDelt)/self.animDuration))*375, colors.WATER_200, 0.0175, 30)
        #SinGraf(self.root, es, 20, 200 - (((time.time() - self.animDelt)/self.animDuration))*300, colors.WATER_300, 0.013, 50)

        Button(self.root, es, self.app.width-150, 150 - ((time.time() - self.animDelt)/self.animDuration)*250, 250, 250, lambda: None, (255, 255, 255), Anchor.RIGHT)
        pg.draw.circle(self.root, (60, 60, 60), (self.app.width-125, 125 - ((time.time() - self.animDelt)/self.animDuration)*225), 225/3, 15)
        pg.draw.rect(self.root, (60, 60, 60), pg.Rect(self.app.width-125-7.5, 125-10 - ((time.time() - self.animDelt)/self.animDuration)*235, 15, 55))
        pg.draw.rect(self.root, (60, 60, 60), pg.Rect(self.app.width-125-7.5, 125-45 - ((time.time() - self.animDelt)/self.animDuration)*270, 15, 15))

        Text(self.root, es, self.app.width//2, 125 - ((time.time() - self.animDelt)/self.animDuration)*225, "Помести банку в банкоприемник", 64, (35, 35, 35), 'Arial', Anchor.CENTER, True)
        
        Text(self.root, es, self.app.width//2, self.app.height//2-100 - ((time.time() - self.animDelt)/self.animDuration)*(self.app.height//2), "Банка ура оооооо да банка ахереть это че банка", 44, (0, 0, 0), 'Arial', Anchor.CENTER)

        if (time.time() - self.animDelt) >= self.animDuration: self.toScreen(CardScreen)

class CardScreen(Screen):
    def __init__(self, *args):
        super().__init__(*args)
        self.app.servo.open()

    def __call__(self):
        self.root.fill((255, 255, 255))
        es = EventState()
        self.call()

        stDrawPointX = (self.app.width // 6)
        stDrawPointY = (self.app.height // 3) * 2
        pg.draw.circle(self.root, (50, 50, 200), (stDrawPointX, stDrawPointY), 40)
        pg.draw.arc(self.root, (50, 50, 200), pg.Rect(stDrawPointX-160, stDrawPointY-160, 160*2, 160*2), 0, math.pi/2, 50)
        pg.draw.arc(self.root, (50, 50, 200), pg.Rect(stDrawPointX-260, stDrawPointY-260, 260*2, 260*2), 0, math.pi/2, 50)
        pg.draw.arc(self.root, (50, 50, 200), pg.Rect(stDrawPointX-360, stDrawPointY-360, 360*2, 360*2), 0, math.pi/2, 50)

        Text(self.root, es, (self.app.width // 2), (self.app.height//3), "Приложи любую электронную", 48, (0, 0, 0), 'Arial', Anchor.LEFT, True)
        Text(self.root, es, (self.app.width // 2), (self.app.height//3) + 100, "карту к считывателю", 48, (0, 0, 0), 'Arial', Anchor.LEFT, True)

        if self.app.rfid.presentedCard()[0]: self.toScreen(CardedScreen)

class CardedScreen(Screen):
    def __init__(self, *args):
        super().__init__(*args)
        self.app.air()

    def __call__(self):
        self.root.fill((255, 255, 255))
        es = EventState()
        self.call()

        stDrawPointX = (self.app.width // 6)
        stDrawPointY = (self.app.height // 3) * 2
        pg.draw.circle(self.root, (50, 50, 200), (stDrawPointX, stDrawPointY), 40)
        pg.draw.arc(self.root, (50, 50, 200), pg.Rect(stDrawPointX-160, stDrawPointY-160, 160*2, 160*2), 0, math.pi/2, 50)
        pg.draw.arc(self.root, (50, 50, 200), pg.Rect(stDrawPointX-260, stDrawPointY-260, 260*2, 260*2), 0, math.pi/2, 50)
        pg.draw.arc(self.root, (50, 50, 200), pg.Rect(stDrawPointX-360, stDrawPointY-360, 360*2, 360*2), 0, math.pi/2, 50)

        pg.draw.polygon(self.root, (180, 180, 180), [[stDrawPointX-50, stDrawPointY-340],
                                                     [stDrawPointX+280, stDrawPointY-290],
                                                     [stDrawPointX+280, stDrawPointY+40],
                                                     [stDrawPointX-50, stDrawPointY]])
        
        Text(self.root, es, (self.app.width // 2), (self.app.height // 3), str(self.app.rfid.presentedCard()[1]), 48, (0, 0, 0), 'Arial', Anchor.LEFT, True)

class App:
    def __init__(self):
        self.logger = Logger()
        self.config = json.load(open("pfand_configs.json"))
        self.logger(f"config loaded, uid: {self.config['machine_id']}")
        self.root = pg.display.set_mode((0, 0), pg.FULLSCREEN)
        self.width = self.root.get_width()
        self.height = self.root.get_height()
        self.bankAnimation = BankAnimation(self.config, self.height//2-100)
        self.bankWorkState = BankWorkState.NOTHING
        self.neural = Neural(self)
        self.storageMode = StorageMode.NOT_SET
        self.hx711 = dvs.HX711(self.config, self.logger)
        self.rfid = dvs.RFID(self.config, self.logger)
        self.servo = dvs.Servo(self.config, self.logger)
        self.air = dvs.AIR(self.config, self.logger)
        if dvs.is_emulator: dvs.createEmulator()
        self.wsclient = WsClient(self.config, self.logger)
        self.screen = InitScreen(self)
        self.logger("App inited")

    def __call__(self):
        self.screen()
        self.neural()
        self.servo()
        pg.display.flip()

    def run_app(self):
        self.delta_time = time.time()
        self.logger("App runned")
        while 1:
            self()
            pg.event.get()

app = App()
app.run_app()