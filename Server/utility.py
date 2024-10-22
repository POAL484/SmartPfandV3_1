def no_object_id(data: dict):
    data.pop("_id")
    return data