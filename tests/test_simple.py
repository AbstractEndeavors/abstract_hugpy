from imports import *
task="text-generation"
prompt = "hi hi hi hi hi hi hi hi hi hi hi hi"
data={"prompt":prompt,"task":task}
input(asyncio.run(execute_prompt(**data)))

