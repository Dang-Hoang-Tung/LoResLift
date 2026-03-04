from transformers import AutoModelForCausalLM, AutoTokenizer


def load_model(model_id: str, hf_token: str = None):
    """Load and return (tokenizer, model) from HuggingFace."""
    tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_token, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        token=hf_token,
        trust_remote_code=True,
    )
    return tokenizer, model
