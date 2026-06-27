from imports import *
from imports.src.abstract_hugpy.flask_app import get_hugpy_flask
app = get_hugpy_flask(name="6092_abstractgpt_api", allowed_origins=["https://dev.abstractgpt.ai/*","https://abstractgpt.ai/*","https://api.abstractgpt.ai/*"],debug=True)

VIDEO_PATH = "/home/op/Videos/Aaron Smith - Dancin (KRONO Remix).mp4"
IMAGE_PATH = "/home/op/Pictures/teragraph.jpg"
PROMPT = f"hi im testing the execute prompt, it includes IMAGE_PATH == {IMAGE_PATH} and VIDEO_PATH == {VIDEO_PATH}"
for model_key,values in MODEL_REGISTRY.items():
    print(f"model_key == {model_key}")
    print(f"values == {values}")
    try:
        input(asyncio.run(execute_prompt(prompt = PROMPT,model=values,video_path=VIDEO_PATH,image_path = IMAGE_PATH,path=VIDEO_PATH,file=VIDEO_PATH,model_key=model_key,)))
    except Exception as e:
        input(f"I errored!! {e}")
