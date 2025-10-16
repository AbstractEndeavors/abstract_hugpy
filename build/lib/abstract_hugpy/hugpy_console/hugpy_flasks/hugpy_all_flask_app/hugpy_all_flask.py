from abstract_flask import *
from abstract_utilities import *
from ...video_console import *
from ...imports import *

hugpy_zerosearch_bp,logger = get_bp('hugpy_zerosearch_bp')
zerosearch = get_zerosearch()

@hugpy_zerosearch_bp.route("/zerosearch_generate", methods=["POST","GET"])
def zerosearchGenerate():
    data = get_request_data(request)
    initialize_call_log(data=data)
    try:        
        if not data:
            return get_json_response(value=f"not prompt in {data}",status_code=400)
        result = zerosearch.generate(**data)
        if not result:
            return get_json_response(value=f"no result for {data}",status_code=400)
        return get_json_response(value=result,status_code=200)
    except Exception as e:
        message = f"{e}"
        return get_json_response(value=message,status_code=500)
