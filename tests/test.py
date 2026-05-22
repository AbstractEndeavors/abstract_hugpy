from imports import *

VIDEO_PATH = "/home/op/Videos/Aaron Smith - Dancin (KRONO Remix).mp4"

# 1) frames
extract = asyncio.run(execute_prompt(file=VIDEO_PATH, capture_frames=True))

# 2) wire vision once, at the edge
runner = VisionRunner(VisionBackendConfig(model_key="Qwen2.5-VL-7B-Instruct"))

# 3) configure the run
cfg = VideoAnalysisConfig(prompt="please provide analysis of this video frame")

# 4) execute
summary = asyncio.run(analyze_video(extract, runner, cfg))
print(summary.model_dump_json(indent=2))
