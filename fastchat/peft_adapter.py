"""PeftModel adapter."""

import os
from peft import PeftConfig, PeftModel
from model_adapter import get_model_adapter
from conversation import Conversation

# Check an environment variable to check if we should be sharing Peft model
# weights.  When false we treat all Peft models as separate.
peft_share_base_weights = (
    os.environ.get("PEFT_SHARE_BASE_WEIGHTS", "false").lower() == "true"
)

peft_model_cache = {}


class PeftModelAdapter:
    """Loads any "peft" model and it's base model."""

    name = "PeftModelAdapter"

    def match(self, model_path: str):
        """Accepts any model path with "peft" in the name"""
        if os.path.exists(os.path.join(model_path, "adapter_config.json")):
            return True
        return "peft" in model_path.lower()

    def load_model(self, model_path: str, from_pretrained_kwargs: dict):
        """Loads the base model then the (peft) adapter weights"""

        config = PeftConfig.from_pretrained(model_path)
        base_model_path = config.base_model_name_or_path
        if "peft" in base_model_path:
            raise ValueError(
                f"PeftModelAdapter cannot load a base model with 'peft' in the name: {config.base_model_name_or_path}"
            )

        # Basic proof of concept for loading peft adapters that share the base
        # weights.  This is pretty messy because Peft re-writes the underlying
        # base model and internally stores a map of adapter layers.
        # So, to make this work we:
        #  1. Cache the first peft model loaded for a given base models.
        #  2. Call `load_model` for any follow on Peft models.
        #  3. Make sure we load the adapters by the model_path.  Why? This is
        #  what's accessible during inference time.
        #  4. In get_generate_stream_function, make sure we load the right
        #  adapter before doing inference.  This *should* be safe when calls
        #  are blocked the same semaphore.
        if peft_share_base_weights:
            if base_model_path in peft_model_cache:
                model, tokenizer = peft_model_cache[base_model_path]
                # Super important: make sure we use model_path as the
                # `adapter_name`.
                model.load_adapter(model_path, adapter_name=model_path)
            else:
                base_adapter = get_model_adapter(base_model_path)
                base_model, tokenizer = base_adapter.load_model(
                    base_model_path, from_pretrained_kwargs
                )
                # Super important: make sure we use model_path as the
                # `adapter_name`.
                model = PeftModel.from_pretrained(
                    base_model, model_path, adapter_name=model_path
                )
                peft_model_cache[base_model_path] = (model, tokenizer)
            return model, tokenizer

        # In the normal case, load up the base model weights again.
        base_adapter = get_model_adapter(base_model_path)
        base_model, tokenizer = base_adapter.load_model(
            base_model_path, from_pretrained_kwargs
        )
        model = PeftModel.from_pretrained(base_model, model_path)
        return model, tokenizer

    def get_default_conv_template(self, model_path: str) -> Conversation:
        """Uses the conv template of the base model"""
        config = PeftConfig.from_pretrained(model_path)
        if "peft" in config.base_model_name_or_path:
            raise ValueError(
                f"PeftModelAdapter cannot load a base model with 'peft' in the name: {config.base_model_name_or_path}"
            )
        
        base_model_path = config.base_model_name_or_path
        base_adapter = get_model_adapter(base_model_path)
        return base_adapter.get_default_conv_template(config.base_model_name_or_path)
