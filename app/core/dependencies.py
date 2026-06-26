from pathlib import Path
from model import Recognizer

_recognizers: dict[str, Recognizer] = {}
train_status: dict = {"state": "idle"}


def get_recognizer(org_id: str) -> Recognizer:
    if org_id not in _recognizers:
        _recognizers[org_id] = Recognizer(Path(f"models/{org_id}_model.pth"))
    return _recognizers[org_id]


def reload_recognizer(org_id: str) -> Recognizer:
    _recognizers[org_id] = Recognizer(Path(f"models/{org_id}_model.pth"))
    return _recognizers[org_id]
