import websockets as ws
from websockets.server import serve
import pymongo as mongodb
import json
from cryptography.fernet import Fernet
import base64 as bs64
import bson
from bson.objectid import ObjectId
import asyncio

from security import *
import ops
from neural import *

class Server:
    def __init__(self, port: int = 9090):
        self.port = port

        self.db_client = mongodb.MongoClient("mongodb://localhost:27017")

        self.db = self.db_client.pfand
        self.machines = self.db.tokens.machines
        self.users = self.db.users
        self.db_info = self.db.info.find_one({"info": "info"})

        self.neural = Neural()

        self.ws_machines = set()
        self.ws_machines_id = []

    async def makeCodeResponse(self, ws, code: int, status: str):
        await ws.send(json.dumps({
            "status": status,
            "code": code
        }))

    async def check_and_return_json(self, ws, data: str):
        try: dataNew = json.loads(data)
        except json.JSONDecodeError:
            await self.makeCodeResponse(ws, 901, "err")
            return None
        if type(dataNew) != type(dict()):
            await self.makeCodeResponse(ws, 901, "err")
            return None
        return dataNew
        
    async def need_fields(self, ws, data: dict, *args, **kwargs):
        if "list" in kwargs.keys():
            for i in kwargs['list']:
                if not i in data.keys():
                    await self.makeCodeResponse(ws, 902, "err")
                    return False
            return True
        else:
            for i in args:
                if not i in data.keys():
                    await self.makeCodeResponse(ws, 902, "err")
                    return False
            return True

    async def make_auth(self, ws_machine):
        await self.makeCodeResponse(ws_machine, 801, "ok")
        about_machine = None
        try: 
            async with asyncio.timeout(60):
                about_machine = await self.check_and_return_json(ws_machine, await ws_machine.recv())
        except TimeoutError: return None
        if not about_machine: return None
        if not await self.need_fields(ws_machine, about_machine, "machine_id", "token"): return None
        if not machine_checker(about_machine['machine_id'], about_machine['token'], self.machines): return None
        refresh_token, machine = new_access_token(about_machine['machine_id'], self.machines)
        self.machines.find_one_and_replace({"machine_id": about_machine['machine_id']}, machine)
        await ws_machine.send(json.dumps({
            "code": 802,
            "status": "ok",
            "data": {
                "token": refresh_token.decode("ascii")
            }
        }))
        return about_machine['machine_id']


    async def newMachineConnected(self, ws_machine):
        machine_id = await self.make_auth(ws_machine)
        if not machine_id:
            await self.makeCodeResponse(ws_machine, 903, "err")
            return
        async for msg in ws_machine:
            data = await self.check_and_return_json(ws_machine, msg)
            if not data: continue
            if not await self.need_fields(ws_machine, data, "op", "data"): continue
            if not data['op'] in ops.OPS:
                await self.makeCodeResponse(ws_machine, 904, "err")
                continue
            if not await self.need_fields(ws_machine, data['data'], list=ops.OPS[data['op']]['fields']): continue
            status, resp_data = await ops.OPS[data['op']]['func'](machine_id, data['data'], self.users, self.machines, self.db_info, self)
            print("op:", data['op'], "status:", status)
            await ws_machine.send(json.dumps({
                "op": "ping",
                "status": "ok",
                "data": "ok"
            }))
            await ws_machine.send(json.dumps({
                "op": data['op'],
                "status": status,
                "data": resp_data
            }))
            await asyncio.sleep(.05)
            await ws_machine.send(json.dumps({
                "op": "ping",
                "status": "ok",
                "data": "ok"
            }))


    async def _start_server(self):
        async with serve(self.newMachineConnected, "0.0.0.0", self.port) as ws_server:
            await asyncio.Future()

    def run(self):
        asyncio.run(self._start_server())

server = Server()
server.run()