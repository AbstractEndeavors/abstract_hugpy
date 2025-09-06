from abstract_flask import *
from abstract_utilities import *
deep_coder_bp,logger = get_bp('deep_coder_bp')
from ..hugging_face_models.deepcoder import *
deepcoder = get_deep_coder()

@deep_coder_bp.route("/deepcoder_generate", methods=["POST","GET"])
def deepcoderGenerate():
    data = get_request_data(request)
    initialize_call_log(data=data)
    try:        
        if not data:
            return get_json_call_response(value=f"not prompt in {data}",status_code=400)
        result = deepcoder.generate(**data)
        if not result:
            return get_json_call_response(value=f"no result for {data}",status_code=400)
        return get_json_response(value=result,status_code=200)
    except Exception as e:
        message = f"{e}"
        return get_json_call_response(value=message,status_code=500)
