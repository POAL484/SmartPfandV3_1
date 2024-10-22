from flask import Flask, request, abort
import pymongo as mongodb
import json
from cryptography.fernet import Fernet
import base64 as bs64
import bson
from bson.objectid import ObjectId

from security import *

app = Flask(__name__)

db_client = mongodb.MongoClient("mongodb://localhost:27017")

db = db_client.pfand
tokens = db.tokens.machines
users = db.users
db_info = db.info.find_one({"info": "info"})

def is_valid_request(request: dict, fields: tuple):
    for i in fields:
        if not i in request.keys():
            return False
    return True

def no_object_id(data: dict):
    data.pop("_id")
    return data

@app.route("/user_update/")
def user_update():
    if not is_valid_request(request.form, ("machine_id", "token", "data")):
        abort(404)
    if not machine_checker(request.form['machine_id'], request.form['token'], tokens):
        abort(400)
    data = json.loads(decrypt(request.form['machine_id'], tokens, bytes(request.form['data'], "utf-8")))
    if users.find_one({"card_uuid": data['card_uuid']}):
        users.find_one_and_replace({"card_uuid": data['card_uuid']}, data)
    else:
        user = data
        user['user_id'] = db_info.last_user_id + 1
        db_info.last_user_id += 1
    refresh_token, machine = new_access_token(request.form['machine_id'], tokens)
    tokens.find_one_and_replace({"machine_id": request.form['machine_id']}, machine)
    return json.dumps(
        {
            "status": "ok",
            "refresh_token": refresh_token.decode("ascii")
        }
    )

@app.route("/user_get/")
def user_get():
    if not is_valid_request(request.form, ("machine_id", "token", "data")):
        abort(400)
    if not machine_checker(request.form['machine_id'], bytes(request.form['token'], "ascii"), tokens):
        abort(403)
    data = json.loads(request.form['data'])
    if users.find_one({"card_uuid": data['card_uuid']}):
        refresh_token, machine = new_access_token(request.form['machine_id'], tokens)
        tokens.find_one_and_replace({"machine_id": request.form['machine_id']}, machine)
        return json.dumps(
            {
                "status": "ok",
                "refresh_token": refresh_token.decode("ascii"),
                "data": encrypt(request.form['machine_id'], tokens, json.dumps(no_object_id(users.find_one({"card_uuid": data['card_uuid']})))).decode("utf-8")
            }
        )
    abort(402)

if __name__ == "__main__":
    app.run("localhost", "9090")