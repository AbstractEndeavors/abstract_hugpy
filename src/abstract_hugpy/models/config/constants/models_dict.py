MODELS = {
	"text_summarization": {
        "include": None,
        "framework": "transformers",
        "task": "summarization",
        "filename": None,
        "hub_id": "Falconsai/text_summarization",
        "folder": "text_summarization",
        "name": "text_summarization"
	    },

    "Qwen2.5-Coder-1.5B-GGUF": {
        "include": "*Q4_K_M.gguf",
        "framework": "llama_cpp",
        "task": "code-generation",
        "filename": "Qwen2.5-Coder-1.5B-Instruct-Q4_K_M.gguf",
        "hub_id": "bartowski/Qwen2.5-Coder-1.5B-Instruct-GGUF",
        "folder": "Qwen/Qwen2.5-Coder-1.5B-GGUF",
        "name": "Qwen2.5-Coder-1.5B-GGUF"
        },
    "DeepCoder-14B": {
        "include": None,
        "framework": "transformers",
        "task": "code-generation",
        "filename": None,
        "hub_id": "agentica-org/DeepCoder-14B-Preview",
        "folder": "DeepCoder-14B",
        "name": "DeepCoder-14B"
        },
    "led_large_16384": {
        "include": None,
        "framework": "transformers",
        "task": "long-summarization",
        "filename": None,
        "hub_id": "allenai/led-large-16384",
        "folder": "led_large_16384",
        "name": "led_large_16384"
        },
    "Qwen3.6-35B-A3B": {
        "include": "*.safetensors",
        "framework": "transformers",
        "task": "vision-language",
        "filename": None,
        "hub_id": "Qwen/Qwen3.6-35B-A3B",
        "folder": "Qwen/Qwen3.6-35B-A3B",
        "name": "Qwen3.6-35B-A3B"
        },
    "all_minilm_l6_v2": {
        "include": None,
        "framework": "transformers",
        "task": "embeddings",
        "filename": None,
        "hub_id": "sentence-transformers/all-MiniLM-L6-v2",
        "folder": "all_minilm_l6_v2",
        "name": "all_minilm_l6_v2"
        },
    "Qwen2.5-Coder-3B-GGUF": {
        "include": "*q4_k_m.gguf",
        "framework": "llama_cpp",
        "task": "code-generation",
        "filename": "qwen2.5-coder-3b-instruct-q4_k_m.gguf",
        "hub_id": "Qwen/Qwen2.5-Coder-3B-Instruct-GGUF",
        "folder": "Qwen/Qwen2.5-Coder-3B-GGUF",
        "name": "Qwen2.5-Coder-3B-GGUF"
        },
    "ZeroSearch_model": {
        "include": None,
        "hub_id": "ZeroSearch_model",
        "folder": "ZeroSearch_model",
        "task": "code-generation",
        "name": "ZeroSearch_model",
        "framework": "transformers",
        "filename": None
        },

    "Qwen3-Coder-Next-Q4_K_M": {
        "include": None,
        "hub_id": "Qwen/Qwen3-Coder-Next-GGUF/Qwen3-Coder-Next-Q4_K_M",
        "folder": "Qwen/Qwen3-Coder-Next-GGUF/Qwen3-Coder-Next-Q4_K_M",
        "task": "code-generation",
        "name": "Qwen3-Coder-Next-Q4_K_M",
        "framework": "llama_cpp",
        "filename": "Qwen3-Coder-Next-Q4_K_M-00001-of-00004.gguf"
        },
    "whisper-large-v3": {
        "include": None,
        "framework": "transformers",
        "task": "speech-recognition",
        "filename": None,
        "hub_id": "openai/whisper-large-v3",
        "folder": "openai/whisper-large-v3",
        "name": "whisper-large-v3"
        },
    "ZeroSearch_dataset": {
        "include": None,
        "framework": "transformers",
        "task": "dataset",
        "filename": None,
        "hub_id": "ZeroSearch/dataset",
        "folder": "ZeroSearch_dataset",
        "name": "ZeroSearch_dataset"
        },

    "whisper-large-v3-turbo": {
        "include": None,
        "framework": "transformers",
        "task": "speech-recognition",
        "filename": None,
        "hub_id": "openai/whisper-large-v3-turbo",
        "folder": "openai/whisper-large-v3-turbo",
        "name": "whisper-large-v3-turbo"
        },
    "flan_t5_xl": {
        "include": None,
        "framework": "transformers",
        "task": "text-generation",
        "filename": None,
        "hub_id": "google/flan-t5-xl",
        "folder": "flan_t5_xl",
        "name": "flan_t5_xl"
        },
    "Qwen2.5-VL-7B-Instruct": {
        "include": None,
        "framework": "transformers",
        "task": "vision-language",
        "filename": None,
        "hub_id": "Qwen/Qwen2.5-VL-7B-Instruct",
        "folder": "Qwen/Qwen2.5-VL-7B-Instruct",
        "name": "Qwen2.5-VL-7B-Instruct"
        }
    }
