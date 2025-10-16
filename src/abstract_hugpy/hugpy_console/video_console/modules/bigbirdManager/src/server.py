# /mnt/24T/hugging_face/new_hugs/keybertManager/server.py
from ..imports import *
summarizer_bp,logger = get_bp('summarizer_bp')
from .manager_utils import (
    generate_with_bigbird,
    refine_with_gpt
)
@summarizer_bp.route("/generate_with_bigbird", methods=["POST"])
def generateWithBigbird():
    args,data = get_args_kwargs(request)
    result = generate_with_bigbird(*args,**data)
    return jsonify({"result": result})
@summarizer_bp.route("/refine_with_gpt", methods=["POST"])
def refineWithGpt():
    args,data = get_args_kwargs(request)
    result = refine_with_gpt(*args,**data)
    return jsonify({"result": result})
