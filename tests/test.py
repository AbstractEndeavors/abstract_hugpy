from imports import *
port = 7000

for module_key,values in MODEL_REGISTRY.items():
    if values.framework == "llama_cpp":
        print(module_key)
        port+=1
        input(values.port)
        
input(sorted(MODEL_REGISTRY.keys()))#asyncio.run(summarize_text('hihihi')))
edit_service  6008_abstractendeavors_qwen25_coder_15b
sudo cp /etc/systemd/system/6008_abstractendeavors_qwen25_coder_15b.service /etc/systemd/system/7000_abstractendeavors_qwen25_coder_15b.service
sudo systemctl stop  6008_abstractendeavors_qwen25_coder_15b
sudo systemctl disable 6008_abstractendeavors_qwen25_coder_15b
sudo systemctl enable 7000_abstractendeavors_qwen25_coder_15b
sudo systemctl start 7000_abstractendeavors_qwen25_coder_15b
