<img src="./LOGO.png"></img>

A library to inspect itermediate layers of PyTorch models.

## Install

```bash
$ pip install surgeon-pytorch
```

[![PyPI - Python Version](https://img.shields.io/pypi/v/surgeon-pytorch?style=flat&colorA=0f0f0f&colorB=0f0f0f)](https://pypi.org/project/surgeon-pytorch/)

## Usage

### Inspect

Given a PyTorch model we can display all layers using `get_layers`:

```python
import torch
import torch.nn as nn

from surgeon_pytorch import Inspect, get_layers

class SomeModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(5, 3)
        self.layer2 = nn.Linear(3, 2)
        self.layer3 = nn.Linear(2, 1)

    def forward(self, x):
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        y = self.layer3(x2)
        return y


model = SomeModel()
print(get_layers(model)) # ['layer1', 'layer2', 'layer3']
```

Then we can wrap our `model` to be inspected using `Inspect` and in every forward call the new model we will also output the provided layer outputs (in second return value):

```python
model_wrapped = Inspect(model, layer='layer2')
x = torch.rand(1, 5)
y, x2 = model_wrapped(x)
print(x2) # tensor([[-0.2726,  0.0910]], grad_fn=<AddmmBackward0>)
```

We can also provide a list of layers:

```python
model_wrapped = Inspect(model, layer=['layer1', 'layer2'])
x = torch.rand(1, 5)
y, [x1, x2] = model_wrapped(x)
print(x1) # tensor([[ 0.1739,  0.3844, -0.4724]], grad_fn=<AddmmBackward0>)
print(x2) # tensor([[-0.2238,  0.0107]], grad_fn=<AddmmBackward0>)
```

Or a dictionary to get named outputs:
```python
model_wrapped = Inspect(model, layer={'x1': 'layer1', 'x2': 'layer2'})
x = torch.rand(1, 5)
y, layers = model_wrapped(x)
print(layers)
"""
{
    'x1': tensor([[ 0.3707,  0.6584, -0.2970]], grad_fn=<AddmmBackward0>),
    'x2': tensor([[-0.1953, -0.3408]], grad_fn=<AddmmBackward0>)
}
"""
```
