import websockets as wbs
import asyncio
import json
import pfand_crypto as sec
import threading as thrd
from enum import Enum
import time

class WsState(Enum):
    DOESNT_CONNECT = 21
    AUTHING = 22
    FAILED = 23
    READY = 24
    MESSAGE = 25
    FAILED_AUTHDATA = 26

class WsClient:
    def __init__(self, config: dict, logger: object):
        self.cfg = config
        self.logger = logger
        self.state = WsState.DOESNT_CONNECT
        self.msg = []
        self.to_send = []
        thrd.Thread(target=self.run_wbs).start()

    def update_config(self):
        json.dump(self.cfg, open("pfand_configs.json", 'w'))

    async def wbs_runner(self):
        self.logger("trying to connect...")
        async for ws in wbs.connect('ws://192.168.53.121:9090'):
            self.logger("connected to ws server")
            self.state = WsState.AUTHING
            msg = json.loads(await ws.recv())
            if msg['status'] != "ok":
                self.logger("first pre-auth message status err, reconect in 5 secs")
                self.state = WsState.FAILED
                await asyncio.sleep(5)
                self.logger("reconnect")
                self.state = WsState.DOESNT_CONNECT
                continue
            await ws.send(json.dumps(
                {"machine_id": self.cfg['machine_id'],
                "token": sec.get_access_token(self.cfg).decode('ascii')}
            ))
            resp = json.loads(await ws.recv())
            if resp['status'] == "err":
                if resp['code'] == 903:
                    self.logger("auth failed, code 903, wrong auth data")
                    self.logger("will idle offline")
                    self.state = WsState.FAILED_AUTHDATA
                    return
                else:
                    self.logger(f"auth failed, code: {resp['code']}  , reconnect in 5 secs")
                    self.state = WsState.FAILED
                    await asyncio.sleep(5)
                    self.logger("reconnect")
                    self.state = WsState.DOESNT_CONNECT
                    continue
            self.logger("auth success")
            self.cfg = sec.set_token(self.cfg, resp['data']['token'])
            self.update_config()
            self.logger("config updated")
            self.state = WsState.READY
            loop = asyncio.new_event_loop()
            thrd.Thread(target=loop.run_forever).start()
            #asyncio.run_coroutine_threadsafe(self.sender(ws), loop)
            thrd.Thread(target=self.sender_start, args=(ws, loop)).start()
            async for msg in ws:
                self.logger(f"message recieved: {msg}")
                self.state = WsState.MESSAGE
                self.msg.append(json.loads(msg))

    def sender_start(self, ws, loop: asyncio.AbstractEventLoop):
        asyncio.run(self.sender(ws))

    async def sender(self, ws):
        print("sender alive!!!!😎")
        while 1:
            if self.to_send:
                send_data = json.dumps(self.to_send.pop(0))
                self.logger(f"sender sends... op: {json.loads(send_data)['op']}")
                #print(send_data)
                #print(len(send_data))
                await ws.send(send_data)
                await asyncio.sleep(.025)

    def run_wbs(self):
        asyncio.run(self.wbs_runner())

    def reping(self, inSecs):
        time.sleep(inSecs)
        self.to_send.append({"op": "ping", "data": {}})

    def neural_call(self, frame):
        #self.to_send.append({"op": "neural.prediction", "data": {"frame": "w"*50}})#, "data": {"frame": frame.tolist()[:30]}})
        frame = str(frame.tolist().copy())
        if len(frame) < 50000:
            self.to_send.append({"op": "neural.prediction.start", "data": {"frame": frame, "num_pcg": len(frame)//50000+1}})
            self.to_send.append({"op": "neural.prediction.finish", "data": {"frame": ""}})
        else:
            self.to_send.append({"op": "neural.prediction.start", "data": {"frame": frame[:50000], "num_pcg": len(frame)//50000+1}})
            frame = frame[50000:]
            while len(frame) > 50000:
                self.to_send.append({"op": "neural.prediction.pcg", "data": {"frame": frame[:50000]}})
                frame = frame[50000:]
            self.to_send.append({"op": "neural.prediction.finish", "data": {"frame": frame}})
        self.to_send.append({"op": "ping", "data": {}})
        thrd.Thread(target=self.reping, args=(5,))

    def read(self):
        if len(self.msg) == 1: self.state = WsState.READY
        return self.msg.pop(0)
    
    def find(self, op):
        for i in range(len(self.msg)):
            if 'op' in self.msg[i].keys():
                if op == self.msg[i]['op']:
                    if len(self.msg) == 1: self.state = WsState.READY
                    return self.msg.pop(i)

if __name__ == "__main__":
    from pfand_types import Logger
    logger = Logger()
    #run_wbs(json.load(open("Client/pfand_configs.json")), logger)