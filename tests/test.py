from imports import *


VIDEO_PATH = "/home/op/Videos/Aaron Smith - Dancin (KRONO Remix).mp4"
IMAGE_PATH="/home/op/Pictures/chandra_bad.jpg"
def test_video_analyzer():
    url = "http://192.168.1.100:7005/analyze"
    
    ##response = postRequest(url)
    extract = asyncio.run(execute_prompt(file=VIDEO_PATH, capture_frames=True))
    # 1) frames
    # 2) wire vision once, at the edge
    runner = runner_for(model_key="Qwen2.5-VL-7B-Instruct")
    # 3) configure the run
    cfg = VideoAnalysisConfig(prompt="please provide analysis of this video frame")
    # 4) execute
    summary = asyncio.run(analyze_video(extract, runner, cfg))
    print(summary.model_dump_json(indent=2))
result = asyncio.run(execute_prompt(model_key="Qwen3-Coder-Next-GGUF", task="text-generation", prompt="hihihi"))
input(result)

