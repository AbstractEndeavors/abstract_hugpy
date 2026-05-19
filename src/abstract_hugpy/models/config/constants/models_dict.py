MODELS = {
  "gte-large-en-v1.5": {
    "model_max_length": 32768,
    "include": None,
    "name": "gte-large-en-v1.5",
    "framework": "transformers",
    "hub_id": "Alibaba-NLP/gte-large-en-v1.5",
    "filename": None,
    "folder": "Alibaba-NLP/gte-large-en-v1.5",
    "task": "code-generation"
  },
  "flan-t5-xl": {
    "model_max_length": 32768,
    "include": None,
    "name": "flan-t5-xl",
    "framework": "transformers",
    "hub_id": "google/flan-t5-xl",
    "filename": None,
    "folder": "google/flan-t5-xl",
    "task": "code-generation"
  },
  "Qwen2.5-VL-7B-Instruct": {
    "model_max_length": 32768,
    "include": None,
    "name": "Qwen2.5-VL-7B-Instruct",
    "framework": "transformers",
    "hub_id": "Qwen/Qwen2.5-VL-7B-Instruct",
    "filename": None,
    "folder": "Qwen/Qwen2.5-VL-7B-Instruct",
    "task": "vision-language"
  },
  "Qwen3-Coder-Next-GGUF": {
    "model_max_length": 32768,
    "include": "Qwen3-Coder-Next-Q4_K_M/*.gguf",
    "name": "Qwen3-Coder-Next-GGUF",
    "framework": "llama_cpp",
    "hub_id": "Qwen/Qwen3-Coder-Next-GGUF",
    "filename": "Qwen3-Coder-Next-Q4_K_M-00001-of-00004.gguf",
    "folder": "Qwen/Qwen3-Coder-Next-GGUF",
    "task": "code-generation"
  },
  "DAN-Qwen3-1.7B": {
    "model_max_length": 32768,
    "include": None,
    "name": "DAN-Qwen3-1.7B",
    "framework": "transformers",
    "hub_id": "UnfilteredAI/DAN-Qwen3-1.7B",
    "filename": None,
    "folder": "UnfilteredAI/DAN-Qwen3-1.7B",
    "task": "text-generation"
  },
  "whisper-large-v3-turbo": {
    "model_max_length": 32768,
    "include": None,
    "name": "whisper-large-v3-turbo",
    "framework": "transformers",
    "hub_id": "openai/whisper-large-v3-turbo",
    "filename": None,
    "folder": "openai/whisper-large-v3-turbo",
    "task": "speech-recognition"
  },
  "flux2-klein-9b-uncensored-text-encoder": {
    "model_max_length": 131072,
    "include": None,
    "name": "flux2-klein-9b-uncensored-text-encoder",
    "framework": "llama_cpp",
    "hub_id": "ponpoke/flux2-klein-9b-uncensored-text-encoder",
    "filename": "flux2-klein-9b-uncensored-f16.gguf",
    "folder": "ponpoke/flux2-klein-9b-uncensored-text-encoder",
    "task": "code-generation"
  },
  "Qwen2.5-Coder-3B-Instruct-GGUF": {
    "model_max_length": 131072,
    "include": None,
    "name": "Qwen2.5-Coder-3B-Instruct-GGUF",
    "framework": "llama_cpp",
    "hub_id": "Qwen/Qwen2.5-Coder-3B-Instruct-GGUF",
    "filename": "qwen2.5-coder-3b-instruct-q4_k_m.gguf",
    "folder": "Qwen/Qwen2.5-Coder-3B-Instruct-GGUF",
    "task": "code-generation"
  },
  "all-minilm-l6-v2": {
    "model_max_length": 32768,
    "include": None,
    "name": "all-minilm-l6-v2",
    "framework": "transformers",
    "hub_id": "sentence-transformers/all-minilm-l6-v2",
    "filename": None,
    "folder": "sentence-transformers/all-minilm-l6-v2",
    "task": "code-generation"
  },
  "text_summarization": {
    "model_max_length": 32768,
    "include": None,
    "name": "text_summarization",
    "framework": "transformers",
    "hub_id": "Falconsai/text_summarization",
    "filename": None,
    "folder": "Falconsai/text_summarization",
    "task": "code-generation"
  },
  "MiniCPM-V-4.6": {
    "model_max_length": 32768,
    "include": None,
    "name": "MiniCPM-V-4.6",
    "framework": "transformers",
    "hub_id": "openbmb/MiniCPM-V-4.6",
    "filename": None,
    "folder": "openbmb/MiniCPM-V-4.6",
    "task": "code-generation"
  },
  "Qwen2.5-Coder-1.5B-Instruct-GGUF": {
    "model_max_length": 32768,
    "include": None,
    "name": "Qwen2.5-Coder-1.5B-Instruct-GGUF",
    "framework": "llama_cpp",
    "hub_id": "bartowski/Qwen2.5-Coder-1.5B-Instruct-GGUF",
    "filename": "Qwen2.5-Coder-1.5B-Instruct-Q4_K_M.gguf",
    "folder": "bartowski/Qwen2.5-Coder-1.5B-Instruct-GGUF",
    "task": "code-generation"
  },
  "led-large-16384": {
    "model_max_length": 32768,
    "include": None,
    "name": "led-large-16384",
    "framework": "transformers",
    "hub_id": "allenai/led-large-16384",
    "filename": None,
    "folder": "allenai/led-large-16384",
    "task": "code-generation"
  },
  "Qwen3.6-35B-A3B": {
    "model_max_length": 32768,
    "include": "*.safetensors",
    "name": "Qwen3.6-35B-A3B",
    "framework": "transformers",
    "hub_id": "Qwen/Qwen3.6-35B-A3B",
    "filename": None,
    "folder": "Qwen/Qwen3.6-35B-A3B",
    "task": "vision-language"
  },
  "whisper-large-v3": {
    "model_max_length": 32768,
    "include": None,
    "name": "whisper-large-v3",
    "framework": "transformers",
    "hub_id": "openai/whisper-large-v3",
    "filename": None,
    "folder": "openai/whisper-large-v3",
    "task": "speech-recognition"
  },
  "Qwen2.5-VL-3B-Instruct": {
    "model_max_length": 32768,
    "include": None,
    "name": "Qwen2.5-VL-3B-Instruct",
    "framework": "transformers",
    "hub_id": "Qwen/Qwen2.5-VL-3B-Instruct",
    "filename": None,
    "folder": "Qwen/Qwen2.5-VL-3B-Instruct",
    "task": "code-generation"
  },
  "Qwen3.6-27B-AEON-Ultimate-Uncensored-GPTQ-Pro-FOEM-4bit-g128": {
    "model_max_length": 262144,
    "include": None,
    "name": "Qwen3.6-27B-AEON-Ultimate-Uncensored-GPTQ-Pro-FOEM-4bit-g128",
    "framework": "transformers",
    "hub_id": "groxaxo/Qwen3.6-27B-AEON-Ultimate-Uncensored-GPTQ-Pro-FOEM-4bit-g128",
    "filename": None,
    "folder": "groxaxo/Qwen3.6-27B-AEON-Ultimate-Uncensored-GPTQ-Pro-FOEM-4bit-g128",
    "task": "code-generation"
  },
  "DeepCoder-14B-Preview": {
    "model_max_length": 32768,
    "include": None,
    "name": "DeepCoder-14B-Preview",
    "framework": "transformers",
    "hub_id": "agentica-org/DeepCoder-14B-Preview",
    "filename": None,
    "folder": "agentica-org/DeepCoder-14B-Preview",
    "task": "code-generation"
  },
  "openai": {
    "model_max_length": 32768,
    "include": None,
    "name": "openai",
    "framework": "transformers",
    "hub_id": "openai",
    "filename": None,
    "folder": "openai",
    "task": "code-generation"
  },
  "DAN-L3-R1-8B-i1-GGUF": {
    "model_max_length": 32768,
    "include": "*Q4_K_M.gguf",
    "name": "DAN-L3-R1-8B-i1-GGUF",
    "framework": "llama_cpp",
    "hub_id": "mradermacher/DAN-L3-R1-8B-i1-GGUF",
    "filename": "DAN-L3-R1-8B.i1-IQ1_M.gguf",
    "folder": "mradermacher/DAN-L3-R1-8B-i1-GGUF",
    "task": "text-generation"
  }
}
