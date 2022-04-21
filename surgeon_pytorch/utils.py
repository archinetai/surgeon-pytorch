from functools import reduce


def exists(val):
    return val is not None


def get_layers(model):
    return [n for n, _ in model.named_modules()][1:]


def get_module(module, alias):
    names = alias.split(sep=".")
    return reduce(getattr, names, module)
