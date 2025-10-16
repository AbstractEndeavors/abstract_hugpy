# /mnt/24T/hugging_face/new_hugs/keybertManager/server.py
from ..imports import *
flan_summarizer_bp,logger = get_bp('flan_summarizer_bp')
from .manager_utils import (
    get_flan_summary,
    FlanManager
)
@flan_summarizer_bp.route("/", methods=["POST"])
def flanSummary():
    args,data = get_args_kwargs(request)
    result = get_flan_summary(*args,**data)
    return jsonify({"result": result})

