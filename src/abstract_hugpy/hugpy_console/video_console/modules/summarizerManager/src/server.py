# /mnt/24T/hugging_face/new_hugs/keybertManager/server.py
from ..imports import *
summarizer_bp,logger = get_bp('summarizer_bp')
from .manager_utils import (
    get_summary
)
@summarizer_bp.route("/summary", methods=["POST"])
def keywords():
    args,data = get_args_kwargs(request)
    result = get_summary(*args,**data)
    return jsonify({"result": result})

