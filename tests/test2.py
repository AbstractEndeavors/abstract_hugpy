from imports import deepcoder_image_analysis
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

deepcoder_image_analysis(IMAGE_PATH)
