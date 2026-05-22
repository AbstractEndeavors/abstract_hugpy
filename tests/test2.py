from imports import *
from abstract_utilities import *

IMAGE_PATH = "/home/op/Pictures/AE.png"

##
##def deepcoder_image_analysis(image_path: str, prompt: str):
##    vision = get_vision_coder()
##
##    return vision.analyze_image(
##        image_path=image_path,
##        prompt=prompt,
##    )
##if __name__ == "__main__":
##    result = deepcoder_image_analysis(
##        image_path=IMAGE_PATH,
##        prompt="Please describe the image.",
##    )
##
##    print(result)
IMAGE_PATH = "/home/op/Pictures/chandra_bad.jpg"
cfg = VisionBackendConfig(
    model_key="Qwen2.5-VL-7B-Instruct",
    host="192.168.1.100",
    port=7005,
    max_tokens=4096,
)
backend = build_backend(cfg)
import base64

with open(IMAGE_PATH, "rb") as f:
    image_b64 = base64.b64encode(f.read()).decode("ascii")

req = VisionRequest(
    request_id="req-abc123",
    model_key="Qwen2.5-VL-7B-Instruct",
    prompt="describe this image",
    image_b64=image_b64,
)
result = asyncio.run(backend.run(req))
if result.error:
    raise RuntimeError(f"server error: {result.error}")
print(result.text)
