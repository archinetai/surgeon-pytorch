from functools import reduce
from typing import Dict, List, Sequence, Union

import torch.nn as nn
from torch import Tensor

from .utils import exists


def get_layers_alias(model):
    return [n for n, _ in model.named_modules()][1:]


def get_module(module, alias):
    names = alias.split(sep=".")
    return reduce(getattr, names, module)


class Layer:
    def __init__(self, module: nn.Module, alias: str, key: str = None):
        self.module = module
        self.alias = alias
        self.key = key
        self.output = None
        self.hook = None

    def __repr__(self):
        has_module = exists(self.module)
        has_hook = exists(self.hook)
        output_shape = self.output.shape if isinstance(self.output, Tensor) else None
        return (
            f"Layer(has_module={has_module}, alias={self.alias}, key={self.key},"
            f" output_shape={output_shape}, has_hook={has_hook}"
        )


def get_layers_modules(
    model: nn.Module, aliases: Union[str, Sequence[str], Dict[str, str]]
) -> List[Layer]:
    layers = []

    if isinstance(aliases, str):
        layers = [Layer(module=get_module(model, aliases), alias=aliases)]
    elif isinstance(aliases, list) or isinstance(aliases, tuple):
        layers = [
            Layer(module=get_module(model, alias), alias=alias) for alias in aliases
        ]
    elif isinstance(aliases, dict):
        layers = [
            Layer(module=get_module(model, alias), key=name, alias=alias)
            for alias, name in aliases.items()
        ]
    else:
        raise TypeError("layer must be str, list, tuple, or dict")

    return layers


class Inspect(nn.Module):
    def __init__(
        self,
        model: nn.Module,
        layer: Union[str, Sequence[str], Dict[str, str]],
        keep_output: bool = True,
    ):
        super().__init__()
        self.model = model
        self.layers = get_layers_modules(model, layer)
        self.layer = layer
        self.keep_output = keep_output

    def register_hooks(self):
        for layer in self.layers:

            def get_hook(layer):
                def hook(module, input, output):
                    layer.output = output

                return hook

            layer.hook = layer.module.register_forward_hook(get_hook(layer))

    def clear_hooks(self):
        for layer in self.layers:
            layer.hook.remove()
            layer.hook = None

    def forward(self, *args, **kwargs):
        # Forward to model and record layers
        try:
            self.register_hooks()
            model_output = self.model(*args, **kwargs)
        finally:
            self.clear_hooks()

        # Retrieve outputs and return
        is_tuple = isinstance(self.layer, tuple)
        if isinstance(self.layer, list) or is_tuple:
            layers_output = [layer.output for layer in self.layers]
            layers_output = tuple(layers_output) if is_tuple else layers_output
            return (model_output, layers_output) if self.keep_output else layers_output
        elif isinstance(self.layer, dict):
            layers_output = {layer.key: layer.output for layer in self.layers}
            return (model_output, layers_output) if self.keep_output else layers_output

        layer_output = self.layers[0].output

        return (model_output, layer_output) if self.keep_output else layer_output
