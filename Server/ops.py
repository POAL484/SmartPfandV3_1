from security import *
from utility import *
#from wsserver import Server
import json

async def machine_user_get(machine_id, data, users, tokens, info, server):
    if users.find_one(data['filter']):
        return "ok", encrypt(machine_id, tokens, json.dumps(no_object_id(users.find_one(data['filter'])))).decode("utf-8")
    
async def machine_user_set(machine_id, data, users, tokens, info, server):
    new_data = json.loads(decrypt(machine_id, tokens, bytes(data['new_data'], 'utf-8')))
    if users.find_one(data['filter']):
        users.find_one_and_replace(data['filter'], new_data)
    else:
        user = new_data
        user['user_id'] = info['last_user_id'] + 1
        info['last_user_id'] += 1
    return "ok", "ok"

async def neural_prediction_start(machine_id, data, users, tokens, info, server):
    #return "ok", server.neural.predict(data['frame'])
    machine_data = tokens.find_one({"machine_id": machine_id})
    machine_data['data']['frame'] = data['frame']
    tokens.find_one_and_replace({"machine_id": machine_id}, machine_data)
    return "ok", "ok"

async def neural_prediction_pcg(machine_id, data, users, tokens, info, server):
    machine_data = tokens.find_one({"machine_id": machine_id})
    machine_data['data']['frame'] += data['frame']
    tokens.find_one_and_replace({"machine_id": machine_id}, machine_data)
    return "ok", "ok"

async def neural_prediction_finish(machine_id, data, users, tokens, info, server):
    frame = tokens.find_one({"machine_id": machine_id})['data']['frame'] + data['frame']
    machine_data = tokens.find_one({"machine_id": machine_id})
    machine_data['data']['lastFrame'] = frame
    machine_data['data']['frame'] = ""
    tokens.find_one_and_replace({"machine_id": machine_id}, machine_data)
    return "ok", server.neural.predict(json.loads(frame))

OPS = {
    "machine.user.get": {
        "fields": ["filter"],
        "func": machine_user_get
    },
    "machine.user.set": {
        "fields": ["filter", "new_data"],
        "func": machine_user_set
    },
    "neural.prediction.start": {
        "fields": ["frame", "num_pcg"],
        "func": neural_prediction_start
    },
    "neural.prediction.pcg": {
        "fields": ["frame"],
        "func": neural_prediction_pcg
    },
    "neural.prediction.finish": {
        "fields": ["frame"],
        "func": neural_prediction_finish
    }
}