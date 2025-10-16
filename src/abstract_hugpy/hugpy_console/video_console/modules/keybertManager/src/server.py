# /mnt/24T/hugging_face/new_hugs/keybertManager/server.py
from ..imports import *

keybert_bp,logger = get_bp('keybert_bp')
from .manager_utils import (
    get_refined_keywords,
    get_keyword_density,
    get_extracted_keywords,
    get_text_keywords
)

@keybert_bp.route("/keywords", methods=["POST"])
def keywords():
    args,data = get_args_kwargs(request)
    result = get_extracted_keywords(*args,**data)
    return jsonify({"result": result})

@keybert_bp.route("/refine", methods=["POST"])
def refine():
    args,data = get_args_kwargs(request)
    result = get_refined_keywords(*args,**data)
    return jsonify(result)

@keybert_bp.route("/density", methods=["POST"])
def density():
    args,data = get_args_kwargs(request)
    result = get_keyword_density(*args,**data)
    return jsonify(result)

@keybert_bp.route("/text_keywords", methods=["POST"])
def textKeywords():
    args,data = get_args_kwargs(request)
    text,keywords = get_text_keywords(*args,**data)
    text_data = {"text":text,"keywords":keywords}
    data.update(text_data)
    result = get_refined_keywords(*args,**data)
    text_data['keywords']=result
    return jsonify(text_data)
