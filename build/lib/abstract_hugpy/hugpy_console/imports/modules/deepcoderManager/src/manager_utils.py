from ..imports import *
from .manager import ZeroSearch
def get_deep_coder(module_path=None,
                   torch_dtype=None,
                   use_quantization=None
                   ):

    zerosearch = ZeroSearch(
        model_dir=module_path,
        use_quantization=use_quantization
        )
    return zerosearch
def try_deep_coder(module_path=None,
                   torch_dtype=None,
                   use_quantization=None):
    # Example usage
    try:
        # Initialize the DeepCoder module
        deepcoder = get_deep_coder(module_path=module_path,
                   torch_dtype=torch_dtype,
                   use_quantization=use_quantization)
        logger.info("DeepCoder logger initialized and active.")
        # Generate text from a prompt
        prompt = "Write a Python function to calculate the factorial of a number."
        generated_text = deepcoder.generate(
            prompt=prompt,
            max_new_tokens=2,
            use_chat_template=False
        )
        print("Generated Text:", generated_text)

        # Generate text using chat template
        messages = [
            {"role": "system", "content": "You are a helpful coding assistant."},
            {"role": "user", "content": "Explain how to implement a binary search in Python."}
        ]
        chat_response = deepcoder.generate(
            prompt=messages,
            max_new_tokens=1000,
            use_chat_template=True
        )
        print("Chat Response:", chat_response)

        # Save output
        deepcoder.save_output(chat_response, "./output/binary_search_explanation.txt")

        # Get model info
        model_info = deepcoder.get_model_info()
        print("Model Info:", model_info)

    except Exception as e:
        logger.error(f"Example usage failed: {str(e)}")

