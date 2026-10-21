"""One-call starter router.

Every question goes to one cheap OpenRouter model. A local run needs
OPENROUTER_API_KEY. The hosted service uses its own key. This file is an
example. Yugos does not import it.
"""

from safetensors.numpy import load_file

from yugos.inference.actions import CallAction, ModelChoice

MODEL = ModelChoice(provider_id="openrouter", model_id="openai/gpt-5-nano", effort=None)


class Router:
    def solve(self, model_input):
        yield CallAction(model=MODEL, input=model_input)


def load_model(weights_path):
    # The pack requires this file. This starter does not read the tensor.
    load_file(weights_path)
    return Router()
