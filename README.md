<img src="./LOGO.png"></img>

A library to inspect and extract intermediate layers of PyTorch models.

### Why?
It's often the case that we want to _inspect_ intermediate layers of PyTorch models without modifying the code. This can be useful to get attention matrices of language models, visualize layer embeddings, or apply a loss function to intermediate layers. Sometimes we want _extract_ subparts of the model and run them independently, either to debug them or to train them separately. All of this can be done with Surgeon without changing one line of the original model.

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

<details>
<summary> <b> Inspect Multiple Layers </b> </summary>
<br>

We can provide a list of layers:

```python
model_wrapped = Inspect(model, layer=['layer1', 'layer2'])
x = torch.rand(1, 5)
y, [x1, x2] = model_wrapped(x)
print(x1) # tensor([[ 0.1739,  0.3844, -0.4724]], grad_fn=<AddmmBackward0>)
print(x2) # tensor([[-0.2238,  0.0107]], grad_fn=<AddmmBackward0>)
```
</details>
     
<details>
<summary> <b> Name Inspected Layer Outputs </b> </summary>
<br>

We can provide a dictionary to get named outputs:
```python
model_wrapped = Inspect(model, layer={'layer1': 'x1', 'layer2': 'x2'})
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
</details>

<details>
<summary> <b> API </b> </summary>
<br>
    
```python
model = Inspect(
    model: nn.Module,
    layer: Union[str, Sequence[str], Dict[str, str]],
    keep_output: bool = True,
)
```
    
</details>


### Extract

Given a PyTorch model we can display all intermediate nodes of the graph using `get_nodes`:

```python
import torch
import torch.nn as nn
from surgeon_pytorch import Extract, get_nodes

class SomeModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(5, 3)
        self.layer2 = nn.Linear(3, 2)
        self.layer3 = nn.Linear(1, 1)

    def forward(self, x):
        x1 = torch.relu(self.layer1(x))
        x2 = torch.sigmoid(self.layer2(x1))
        y = self.layer3(x2).tanh()
        return y

model = SomeModel()
print(get_nodes(model)) # ['x', 'layer1', 'relu', 'layer2', 'sigmoid', 'layer3', 'tanh']
```

Then we can extract outputs using `Extract`, which will create a new model that returns the requested output node:

```python
model_ext = Extract(model, node_out='sigmoid')
x = torch.rand(1, 5)
sigmoid = model_ext(x)
print(sigmoid) # tensor([[0.5570, 0.3652]], grad_fn=<SigmoidBackward0>)
```

We can also extract a model with new input nodes:

```python
model_ext = Extract(model, node_in='layer1', node_out='sigmoid')
layer1 = torch.rand(1, 3)
sigmoid = model_ext(layer1)
print(sigmoid) # tensor([[0.5444, 0.3965]], grad_fn=<SigmoidBackward0>)
```

<details>
<summary> <b> Multiple Nodes </b> </summary>
<br>    
    
We can also provide multiple inputs and outputs and name them:

```python
model_ext = Extract(model, node_in={ 'layer1': 'x' }, node_out={ 'sigmoid': 'y1', 'relu': 'y2'})
out = model_ext(x = torch.rand(1, 3))
print(out)
"""
{
    'y1': tensor([[0.4437, 0.7152]], grad_fn=<SigmoidBackward0>),
    'y2': tensor([[0.0555, 0.9014, 0.8297]]),
}
"""
```
    
</details>

    
<details>
<summary> <b> Graph Input/Output Summary </b> </summary>
<br> 
    
Note that changing an input node might not be enough to cut the graph (there might be other dependencies connected to previous inputs). To view all inputs of the new graph we can call `model_ext.summary` which will give us an overview of all required inputs and returned outputs:

```python
import torch
import torch.nn as nn
from surgeon_pytorch import Extract, get_nodes

class SomeModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.layer1a = nn.Linear(2, 2)
        self.layer1b = nn.Linear(2, 2)
        self.layer2 = nn.Linear(2, 1)

    def forward(self, x):
        a = self.layer1a(x)
        b = self.layer1b(x)
        c = torch.add(a, b)
        y = self.layer2(c)
        return y

model = SomeModel()
print(get_nodes(model)) # ['x', 'layer1a', 'layer1b', 'add', 'layer2']

model_ext = Extract(model, node_in = {'layer1a': 'my_input'}, node_out = {'add': 'my_add'})
print(model_ext.summary) # {'input': ('x', 'my_input'), 'output': {'my_add': add}}

out = model_ext(x = torch.rand(1, 2), my_input = torch.rand(1,2))
print(out) # {'my_add': tensor([[ 0.3722, -0.6843]], grad_fn=<AddBackward0>)}
```

</details>
    
<details>
<summary> <b> API </b> </summary>
<br> 

#### API

```python
model = Extract(
    model: nn.Module,
    node_in: Optional[Union[str, Sequence[str], Dict[str, str]]] = None,
    node_out: Optional[Union[str, Sequence[str], Dict[str, str]]] = None,
    tracer: Optional[Type[Tracer]] = None,          # Tracer class used, default: torch.fx.Tracer
    concrete_args: Optional[Dict[str, Any]] = None, # Tracer concrete_args, default: None
    keep_output: bool = None,                       # Set to `True` to return original outputs as first argument, default: True except if node_out are provided
    share_modules: bool = False,                    # Set to true if you want to share module weights with original model
)
```

</details>


### Inspect vs Extract
The `Inspect` class always executes the entire model provided as input, and it uses special hooks to record the tensor values as they flow through. This approach has the advantages that (1) we don't create a new module (2) it allows for a dynamic execution graph (i.e. `for` loops and `if` statements that depend on inputs). The downsides of `Inspect` are that (1) if we only need to execute part of the model some computation is wasted, and (2) we can only output values from `nn.Module` layers – no intermediate function values.

The `Extract` class builds an entirely new model using symbolic tracing. The advantages of this approach are (1) we can crop the graph anywhere and get a new model that computes only that part, (2) we can extract values from intermediate functions (not only layers), and (3) we can also change input tensors. The downside of `Extract` is that only static graphs are allowed (note that most models have static graphs).






## TODO
- [x] add extract function to get intermediate block
- [x] add model inputs/outputs summary
