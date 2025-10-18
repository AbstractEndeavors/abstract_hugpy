# /mnt/24T/hugging_face/new_hugs/keybertManager/server.py
from ..imports import *
from .manager_utils import (
    get_deepcoder
)
hugpy_deepcoder_bp,logger = get_bp('hugpy_deepcoder_bp')
deepcoder = get_deepcoder()
@hugpy_deepcoder_bp.route("/deepcoder_generate", methods=["POST","GET"])
def deepcoderGenerate():
    data = get_request_data(request)
    initialize_call_log(data=data)
    try:        
        if not data:
            return get_json_response(value=f"not prompt in {data}",status_code=400)
        result = deepcoder.generate(**data)
        if not result:
            return get_json_response(value=f"no result for {data}",status_code=400)
        return get_json_response(value=result,status_code=200)
    except Exception as e:
        message = f"{e}"
        return get_json_response(value=message,status_code=500)
