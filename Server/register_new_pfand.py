url_safe = list(range(33, 123))
url_safe.remove(47)
url_safe.remove(91)
url_safe.remove(92)
url_safe.remove(93)
url_safe.remove(60)
url_safe.remove(62)
len(url_safe)

from uuid import uuid4
import pymongo as mondodb
from random import choice

mongo_client = mondodb.MongoClient("mongodb://localhost:27017")

db = mongo_client.pfand

tokens = db.tokens.machines

uid = uuid4()
service_token = ''
access_token_unencrypted = ''
for i in range(32):
    service_token += chr(choice(url_safe))
    access_token_unencrypted += chr(choice(url_safe))

tokens.insert_one({
    "machine_id": str(uid),
    "service_token": service_token,
    "access_token": access_token_unencrypted,
    "data": {
        "frame": "",
        "lastFrame": ""
    }
})

print(f"""
    New machine registred
    uid: {uid}
    service_token: {service_token}
    access_token: {access_token_unencrypted}
""")