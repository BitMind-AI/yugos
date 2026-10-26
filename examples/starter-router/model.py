"""Fixed-model interface example for local evaluation.

Every turn selects one OpenRouter model. Replace the constant choice with
learned inference before competitive submission. Yugos does not import this
example as part of its package.
"""

from safetensors.numpy import load_file

from yugos.inference.actions import ModelChoice

MODEL = ModelChoice(provider_id="openrouter", model_id="openai/gpt-5-nano", effort=None)


class Router:
    def route(self, model_input):
        return MODEL


def load_model(weights_path):
    # Validate the required tensor asset; this fixed baseline does not use it.
    load_file(weights_path)
    return Router()
