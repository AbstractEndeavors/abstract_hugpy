import uuid
from typing import *
def get_messages(prompt: str) -> List[dict]:
    return [{"role": "user", "content": prompt}]


def get_request_id() -> str:
    return str(uuid.uuid1())
