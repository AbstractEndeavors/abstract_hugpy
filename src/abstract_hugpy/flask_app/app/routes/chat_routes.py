from ..functions import *
chat_bp,logger=get_bp("chat_router",__name__)
chat_funcs = {
    "chat":{
        "stream":
        chat_stream
        }
    }
register_categories(chat_bp, chat_funcs)

