from abstract_flask import *
from abstract_utilities import *
from ...video_console import *
from ...imports import *

hugpy_deepzero_bp,logger = get_bp('hugpy_deepzero_bp')
def deepZeroGenerate(request,name):
    data = get_request_data(request)
    initialize_call_log(data=data)

    if not data:
        return get_json_call_response(value=f"not prompt in {data}",status_code=400)
    data['name'] = data.get('name') or name
    result = deep_zero_generate(**data)
    if not result:
        return get_json_response(value=f"no result for {data}",status_code=400)
    return get_json_call_response(value=result,status_code=200)

@hugpy_deepzero_bp.route("/deepcoder_generate", methods=["POST","GET"])
def deepcoderGenerate():
    return deepZeroGenerate(request,"deepcoder")

@hugpy_deepzero_bp.route("/zerosearch_generate", methods=["POST","GET"])
def zerosearchGenerate():
    return deepZeroGenerate(request,"zerosearch")
