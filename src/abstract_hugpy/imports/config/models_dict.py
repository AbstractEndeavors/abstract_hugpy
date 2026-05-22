MODELS =  {

  "Qwen2.5-Coder-1.5B-Instruct-GGUF": {
    "model_max_length": 32768,
    "include": None,
    "name": "Qwen2.5-Coder-1.5B-Instruct-GGUF",
    "framework": "llama_cpp",
    "hub_id": "bartowski/Qwen2.5-Coder-1.5B-Instruct-GGUF",
    "filename": "Qwen2.5-Coder-1.5B-Instruct-Q4_K_M.gguf",
    "folder": "bartowski/Qwen2.5-Coder-1.5B-Instruct-GGUF",
    "task": "code-generation",
    "port": 7000
  },
  "Qwen2.5-VL-3B-Instruct": {
    "model_max_length": 32768,
    "include": None,
    "name": "Qwen2.5-VL-3B-Instruct",
    "framework": "transformers",
    "hub_id": "Qwen/Qwen2.5-VL-3B-Instruct",
    "filename": None,
    "folder": "Qwen/Qwen2.5-VL-3B-Instruct",
    "task": "code-generation",
    "port": None
  },
  "Qwen2.5-Coder-3B-Instruct-GGUF": {
    "model_max_length": 131072,
    "include": None,
    "name": "Qwen2.5-Coder-3B-Instruct-GGUF",
    "framework": "llama_cpp",
    "hub_id": "Qwen/Qwen2.5-Coder-3B-Instruct-GGUF",
    "filename": "qwen2.5-coder-3b-instruct-q4_k_m.gguf",
    "folder": "Qwen/Qwen2.5-Coder-3B-Instruct-GGUF",
    "task": "code-generation",
    "port":7001
  },
  "Qwen2.5-VL-7B-Instruct": {
    "model_max_length": 32768,
    "include": None,
    "name": "Qwen2.5-VL-7B-Instruct",
    "framework": "transformers",
    "hub_id": "Qwen/Qwen2.5-VL-7B-Instruct",
    "filename": None,
    "folder": "Qwen/Qwen2.5-VL-7B-Instruct",
    "task": "vision-language",
    "port": 7005
  },
  "DAN-Qwen3-1.7B": {
    "model_max_length": 32768,
    "include": None,
    "name": "DAN-Qwen3-1.7B",
    "framework": "transformers",
    "hub_id": "UnfilteredAI/DAN-Qwen3-1.7B",
    "filename": None,
    "folder": "UnfilteredAI/DAN-Qwen3-1.7B",
    "task": "text-generation",
    "port": None
  },
  "Qwen3.6-27B-AEON-Ultimate-Uncensored-GPTQ-Pro-FOEM-4bit-g128": {
    "model_max_length": 262144,
    "include": None,
    "name": "Qwen3.6-27B-AEON-Ultimate-Uncensored-GPTQ-Pro-FOEM-4bit-g128",
    "framework": "transformers",
    "hub_id": "groxaxo/Qwen3.6-27B-AEON-Ultimate-Uncensored-GPTQ-Pro-FOEM-4bit-g128",
    "filename": None,
    "folder": "groxaxo/Qwen3.6-27B-AEON-Ultimate-Uncensored-GPTQ-Pro-FOEM-4bit-g128",
    "task": "code-generation",
    "port": None
  },
  "Qwen3.6-35B-A3B": {
    "model_max_length": 32768,
    "include": "*.safetensors",
    "name": "Qwen3.6-35B-A3B",
    "framework": "transformers",
    "hub_id": "Qwen/Qwen3.6-35B-A3B",
    "filename": None,
    "folder": "Qwen/Qwen3.6-35B-A3B",
    "task": "vision-language",
    "port": None
  },
  "Qwen3-Coder-Next-GGUF": {
    "model_max_length": 32768,
    "include": "Qwen3-Coder-Next-Q4_K_M/*.gguf",
    "name": "Qwen3-Coder-Next-GGUF",
    "framework": "llama_cpp",
    "hub_id": "Qwen/Qwen3-Coder-Next-GGUF",
    "filename": "Qwen3-Coder-Next-Q4_K_M-00001-of-00004.gguf",
    "folder": "Qwen/Qwen3-Coder-Next-GGUF",
    "task": "code-generation",
    "port": 7002
  },
  "flux2-klein-9b-uncensored-text-encoder": {
    "model_max_length": 131072,
    "include": None,
    "name": "flux2-klein-9b-uncensored-text-encoder",
    "framework": "llama_cpp",
    "hub_id": "ponpoke/flux2-klein-9b-uncensored-text-encoder",
    "filename": "flux2-klein-9b-uncensored-f16.gguf",
    "folder": "ponpoke/flux2-klein-9b-uncensored-text-encoder",
    "task": "code-generation",
    "port": 7003
  },
  "DeepCoder-14B-Preview": {
    "model_max_length": 32768,
    "include": None,
    "name": "DeepCoder-14B-Preview",
    "framework": "transformers",
    "hub_id": "agentica-org/DeepCoder-14B-Preview",
    "filename": None,
    "folder": "agentica-org/DeepCoder-14B-Preview",
    "task": "code-generation",
    "port": None
  },

  "DAN-L3-R1-8B-i1-GGUF": {
    "model_max_length": 32768,
    "include": "*Q4_K_M.gguf",
    "name": "DAN-L3-R1-8B-i1-GGUF",
    "framework": "llama_cpp",
    "hub_id": "mradermacher/DAN-L3-R1-8B-i1-GGUF",
    "filename": "DAN-L3-R1-8B.i1-IQ1_M.gguf",
    "folder": "mradermacher/DAN-L3-R1-8B-i1-GGUF",
    "task": "text-generation",
    "port": 7004
  }
}

