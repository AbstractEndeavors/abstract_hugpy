from ..imports import *
from .manager import ZeroSearch

logger = get_logFile("zerosearch_interface")

# --------------------------------------------------------------------------
# Factory
# --------------------------------------------------------------------------
def get_zerosearch(model_dir: str = None, use_quantization: bool = None):
    """
    Retrieve or initialize the persistent ZeroSearch singleton.

    Args:
        model_dir (str, optional): Path to local model directory or HF repo ID.
        use_quantization (bool, optional): Enable 4-bit quantization if available.

    Returns:
        ZeroSearch: Initialized singleton instance.
    """
    try:
        zerosearch = ZeroSearch(
            model_dir=model_dir,
            use_quantization=use_quantization,
        )
        logger.info("ZeroSearch manager initialized and active.")
        return zerosearch
    except Exception as e:
        logger.error(f"Failed to initialize ZeroSearch: {e}")
        raise


# --------------------------------------------------------------------------
# Smoke test / Example usage
# --------------------------------------------------------------------------
def try_zerosearch(model_dir: str = None, use_quantization: bool = None):
    """
    Quick functional test to verify ZeroSearch model and pipeline behavior.
    """
    try:
        zerosearch = get_zerosearch(
            model_dir=model_dir,
            use_quantization=use_quantization,
        )

        logger.info("ZeroSearch smoke test starting...")

        # Basic text generation
        prompt = "Write a concise summary of what a transformer model is."
        output = zerosearch.generate(
            prompt=prompt,
            max_new_tokens=200,
            temperature=0.7,
            top_p=0.9,
        )
        print("\n--- ZeroSearch Output ---\n", output)

        # Chat-style example
        messages = [
            {"role": "system", "content": "You are a precise and factual assistant."},
            {"role": "user", "content": "Explain how gradient descent works."},
        ]
        chat_output = zerosearch.generate(
            prompt="",
            messages=messages,
            use_chat_template=True,
            max_new_tokens=300,
        )
        print("\n--- ZeroSearch Chat Response ---\n", chat_output)

        # Show environment info
        info = zerosearch.get_info()
        print("\n--- ZeroSearch Info ---\n", info)

        logger.info("ZeroSearch test completed successfully.")

    except Exception as e:
        logger.error(f"ZeroSearch test failed: {e}")
        raise
