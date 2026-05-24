from imports import *
prompt = """Google appears to be running into some hiccups after the company began rolling out its updated, and even more AI-focused search experience at I/O 2026. Currently, searching for the words "disregard," "stop" or "ignore" on Google no longer displays a snippet with a definition, and instead offers an AI Overview and a lot of blank space. Because users have complained about the issue on social media, and publications like TechCrunch and Macrumors have reported on it, even if you don't get a definition, you might still get a collection of links to articles documenting the issue before the traditional list of links.

Multiple members of Engadget's staff were able to recreate the strange AI Overview responses with their own personal Google searches. In Incognito Mode, Google responded correctly once by displaying its usual snippet with the definition, and failed a second time by once again responding with an AI Overview. Links to online dictionaries still appear under these incorrect results, but you have to scroll past an AI Overview or a grid of articles to actually get to them.

Read More: https://www.engadget.com/2179762/google-is-currently-struggling-to-define-words-like-disregard-stop-and-ignore/
"""
tasks = ['any-to-any', 'audio-classification', 'automatic-speech-recognition', 'depth-estimation', 'document-question-answering', 'feature-extraction', 'fill-mask', 'image-classification', 'image-feature-extraction', 'image-segmentation', 'image-text-to-text', 'keypoint-matching', 'mask-generation', 'ner', 'object-detection', 'sentiment-analysis', 'table-question-answering', 'text-classification', 'text-generation', 'text-to-audio', 'text-to-speech', 'token-classification', 'video-classification', 'zero-shot-audio-classification', 'zero-shot-classification', 'zero-shot-image-classification', 'zero-shot-object-detection']
for task in tasks:
    input(asyncio.run(execute_prompt(prompt=prompt,task = task)))
