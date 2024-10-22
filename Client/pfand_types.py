import pygame as pg
import keyboard as kb
import mouse
import pyautogui as pag
from enum import Enum
import math
import time
import json
import datetime as dt

def is_collision(x, y, width, height, mx, my, offset):
    if mx > x - offset and mx < x + width + offset and my > y - offset and my < y + height + offset: return True
    return False

class EventState:
    def __init__(self):
        self.mouseX = pag.position()[0]
        self.mouseY = pag.position()[1]
        self.pressed = pg.mouse.get_pressed()

class Anchor(Enum):
    LEFT = 10
    CENTER = 11
    RIGHT = 12

class StorageMode(Enum):
    NOT_SET = 31
    OFFLINE = 32
    ONLINE = 33

class Text:
    def __init__(self, root: pg.Surface, eventstate: EventState,
                 x: int, y: int, text: str, size: int, color: tuple, font: str, anchor: Anchor, bold: bool = False, italic: bool = False):
        sfont = pg.font.SysFont(font, size, bold, italic)
        render = sfont.render(text, True, color)
        match anchor:
            case Anchor.LEFT:
                root.blit(render, (x, y))
            case Anchor.CENTER:
                root.blit(render, (x-(sfont.size(text)[0]/2), y-(sfont.size(text)[1]/2)))
            case Anchor.RIGHT:
                root.blit(render, (x-sfont.size(text)[0], y-sfont.size(text)[1]))

class Button:
    def __init__(self, root: pg.Surface, eventstate: EventState, x: int, y: int, width: int, height: int, func: callable, color: tuple, anchor: Anchor):
        surf = pg.Surface((width, height))
        surf.fill(color)
        match anchor:
            case Anchor.LEFT: pass
            case Anchor.CENTER:
                x -= width // 2
                y -= height // 2
            case Anchor.RIGHT:
                x -= width
                y -= height
        root.blit(
            surf,
            pg.Rect((x, y, width, height))
        )
        if is_collision(x, y, width, height, eventstate.mouseX, eventstate.mouseY, 30) and eventstate.pressed: func()

class SinGraf:
    def __init__(self, root: pg.Surface, eventstate: EventState,
                 ycenter: int, height: int, color: tuple, kx: float, speed: float):
        '''for i in range(root.get_width()):
            y = int((ycenter * math.sin((i-(time.time()*speed))*kx))+root.get_height()-height)
            surf = pg.Surface((1, height+y))
            surf.fill(color)
            root.blit(
                surf,
                pg.Rect((i, y, 1, height+y))
            )'''
        pass

class Bank:
    @classmethod
    def calcPoint(self, x: int, y: int, angle: int):
        R = math.sqrt(x**2 + y**2)
        stAngle = math.degrees(math.asin(x/R))
        return (R*math.sin(math.radians(stAngle+angle)), R*math.cos(math.radians(stAngle-angle)))


    def __init__(self, root: pg.Surface, eventstate: EventState,
                 xcenter: int, ycenter: int, ksize: float, xcolor: int, rotate: int):
        pg.draw.polygon(root, (180+xcolor, 180+xcolor, 180+xcolor), [
            [xcenter-(self.calcPoint(25, 70, rotate)[0]*ksize), ycenter-(self.calcPoint(25, 70, rotate)[1]*ksize)],
            [xcenter-(self.calcPoint(40, 55, rotate)[0]*ksize), ycenter-(self.calcPoint(40, 55, rotate)[1]*ksize)],
            [xcenter+(self.calcPoint(40, 55, rotate)[0]*ksize), ycenter-(self.calcPoint(40, 55, rotate)[1]*ksize)],
            [xcenter+(self.calcPoint(25, 70, rotate)[0]*ksize), ycenter-(self.calcPoint(25, 70, rotate)[1]*ksize)]])
        pg.draw.polygon(root, (140+xcolor, 140+xcolor, 140+xcolor), [
            [xcenter-(self.calcPoint(40, 55, rotate)[0]*ksize), ycenter-(self.calcPoint(40, 55, rotate)[1]*ksize)],
            [xcenter-(self.calcPoint(40, 55, rotate)[0]*ksize), ycenter+(self.calcPoint(40, 55, rotate)[1]*ksize)],
            [xcenter+(self.calcPoint(40, 55, rotate)[0]*ksize), ycenter+(self.calcPoint(40, 55, rotate)[1]*ksize)],
            [xcenter+(self.calcPoint(40, 55, rotate)[0]*ksize), ycenter-(self.calcPoint(40, 55, rotate)[1]*ksize)]
        ])
        pg.draw.polygon(root, (180+xcolor, 180+xcolor, 180+xcolor), [
            [xcenter-(self.calcPoint(40, 55, rotate)[0]*ksize), ycenter+(self.calcPoint(40, 55, rotate)[1]*ksize)],
            [xcenter-(self.calcPoint(25, 70, rotate)[0]*ksize), ycenter+(self.calcPoint(25, 70, rotate)[1]*ksize)],
            [xcenter+(self.calcPoint(25, 70, rotate)[0]*ksize), ycenter+(self.calcPoint(25, 70, rotate)[1]*ksize)],
            [xcenter+(self.calcPoint(40, 55, rotate)[0]*ksize), ycenter+(self.calcPoint(40, 55, rotate)[1]*ksize)]
        ])
        
class AnimationState(Enum):
    NONE = 41
    READY = 42
    RISE = 43
    FALL = 44

class BankAnimation:
    
    SIZE_TIME = 1.4
    COLOR_TIME = 1.4

    def __init__(self, cfg: dict, yPos: int):
        self.deltaSizeDraw = time.time()
        self.deltaColorDraw = time.time()
        self.deltaSizeCount = time.time()
        self.deltaColorCount = time.time()

        self.sizeAnimation = AnimationState.NONE
        self.colorAnimation = AnimationState.NONE

        self.min_weight = cfg['min_bank_weight']
        self.max_weight = cfg['max_bank_weight']

        self.yPos = yPos

    def __call__(self, weight: float, root, es: EventState):
        size = .05
        xPos = -100
        color = 0
        if weight > self.min_weight:
            if self.sizeAnimation == AnimationState.NONE:
                self.sizeAnimation = AnimationState.RISE
                self.deltaSizeDraw = time.time()
                self.deltaSizeCount = time.time()
            elif self.sizeAnimation == AnimationState.FALL:
                self.sizeAnimation = AnimationState.RISE
                self.deltaSizeCount = time.time() - (self.SIZE_TIME - (time.time() - self.deltaSizeDraw))
                self.deltaSizeDraw = time.time() - (self.SIZE_TIME - (time.time() - self.deltaSizeDraw))
            if self.sizeAnimation != AnimationState.READY:
                size = 1.6 + math.sin((time.time() - self.deltaSizeDraw)*(math.pi/2/self.SIZE_TIME))
                xPos = -75 + math.sin((time.time() - self.deltaSizeDraw)*(math.pi/2/self.SIZE_TIME))*250
            else:
                size = 2.6
                xPos = 175
            if self.sizeAnimation == AnimationState.RISE and time.time() - self.deltaSizeCount >= self.SIZE_TIME:
                self.sizeAnimation = AnimationState.READY
        if weight > self.max_weight:
            if self.colorAnimation == AnimationState.NONE:
                self.colorAnimation = AnimationState.RISE
                self.deltaColorDraw = time.time()
                self.deltaColorCount = time.time()
            elif self.colorAnimation == AnimationState.FALL:
                self.colorAnimation = AnimationState.RISE
                self.deltaColorCount = time.time() - (self.COLOR_TIME - (time.time() - self.deltaColorDraw))
                self.deltaColorDraw = time.time() - (self.COLOR_TIME - (time.time() - self.deltaColorDraw))
            if self.colorAnimation == AnimationState.RISE and time.time() - self.deltaColorCount >= self.COLOR_TIME:
                self.colorAnimation = AnimationState.READY
        else:
            if self.colorAnimation == AnimationState.READY:
                self.colorAnimation = AnimationState.FALL
                self.deltaColorCount = time.time()
                self.deltaColorDraw = time.time()
            elif self.colorAnimation == AnimationState.RISE:
                self.colorAnimation = AnimationState.FALL
                self.deltaColorCount = time.time() - (self.COLOR_TIME - (time.time() - self.deltaColorDraw))
                self.deltaColorDraw = time.time() - (self.COLOR_TIME - (time.time() - self.deltaColorDraw))
            if self.colorAnimation == AnimationState.FALL and time.time() - self.deltaColorCount >= self.COLOR_TIME:
                self.colorAnimation = AnimationState.NONE
        if weight < self.min_weight:
            if self.sizeAnimation == AnimationState.READY:
                self.sizeAnimation = AnimationState.FALL
                self.deltaSizeDraw = time.time()
                self.deltaSizeCount = time.time()
            elif self.sizeAnimation == AnimationState.RISE:
                self.sizeAnimation = AnimationState.FALL
                self.deltaSizeCount = time.time() - (self.SIZE_TIME - (time.time() - self.deltaSizeDraw))
                self.deltaSizeDraw = time.time() - (self.SIZE_TIME - (time.time() - self.deltaSizeDraw))
            if self.sizeAnimation == AnimationState.FALL and time.time() - self.deltaSizeCount >= self.SIZE_TIME:
                self.sizeAnimation = AnimationState.NONE
            if self.sizeAnimation != AnimationState.NONE:
                size = 1.6 + math.sin(math.pi/2 - (time.time() - self.deltaSizeDraw)*(math.pi/2/self.SIZE_TIME))
                xPos = -75 + math.sin(math.pi/2 - (time.time() - self.deltaSizeDraw)*(math.pi/2/self.SIZE_TIME))*250
        if self.colorAnimation == AnimationState.NONE: color = 0
        elif self.colorAnimation == AnimationState.READY: color = -75
        elif self.colorAnimation == AnimationState.RISE: color = math.sin((time.time() - self.deltaColorDraw)*(math.pi/2/self.COLOR_TIME))*-75
        elif self.colorAnimation == AnimationState.FALL: color = math.sin(math.pi/2 - (time.time() - self.deltaColorDraw)*(math.pi/2/self.COLOR_TIME))*-75
        Bank(root, es, xPos, self.yPos, size, color, 0)
            
class BankWorkState(Enum):
    NOTHING = 51
    NEED_START_NEURAL = 52
    NEURAL_CHECK = 53
    NEURAL_FAIL = 54
    CARD = 55

class Logger:
    def __init__(self):
        self.logs = []
        with open("logs.txt", 'w') as saveLogs:
            saveLogs.close()

    def __call__(self, data: str):
        datatime = dt.datetime.now().strftime("%H:%M:%S ") + data
        self.logs.append(datatime)
        with open("logs.txt", "a") as saveLogs:
            saveLogs.write("\n" + datatime)
            saveLogs.close()