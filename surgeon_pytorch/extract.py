import copy
from typing import Any, Dict, List, Optional, Sequence, Type, Union

import torch.nn as nn
from torch.fx import Graph, GraphModule, Tracer

from .utils import exists, to_list


def get_rename_tracer(base: Type[Tracer]):
    """Returns RenameTracer class with custom Tracer parent"""
    base_any: Any = base

    class RenameTracer(base_any):
        """Renames all nodes with module aliases as prefix"""

        def __init__(self):
            super().__init__()
            self.module_alias: str = ""
            self.module_ops_count: Dict[str, int] = {}

        def call_module(self, m: nn.Module, forward, args, kwargs):
            # Save current module alias prefix before entering module
            alias = self.module_alias
            ops_count = self.module_ops_count
            try:
                self.module_alias = self.path_of_module(m)
                self.module_ops_count = {}
                return super().call_module(m, forward, args, kwargs)
            finally:
                # Pop it back
                self.module_alias = alias
                self.module_ops_count = ops_count

        def create_proxy(self, kind, target, args, kwargs, name=None, type_expr=None):
            proxy = super().create_proxy(kind, target, args, kwargs, name, type_expr)
            node = proxy.node
            # Change node names for functions/methods/module with alias
            if node.op == "call_method" or node.op == "call_function":
                if self.module_alias:
                    node.alias = self.module_alias + "." + self.get_name(node)
                else:
                    node.alias = self.get_name(node)
            elif node.op == "call_module":
                node.alias = self.module_alias
            elif node.op == "placeholder":
                node.alias = node.name
            return proxy

        def get_name(self, node):
            """Gets node name with incremental id"""
            # e.g. random_op_2 -> random_op
            name = node.name.rsplit("_", 1)[0]
            # get count
            count = self.module_ops_count.get(name, 0)
            # increment count
            self.module_ops_count[name] = count + 1
            return name + "_" + str(count) if count else name

    return RenameTracer


# By default: class RenameTracer(Tracer)
RenameTracer = get_rename_tracer(Tracer)


def set_tracer(tracer: Type[Tracer]):
    """Change default tracer parent globally"""
    global RenameTracer
    RenameTracer = get_rename_tracer(tracer)


def reset_tracer():
    global RenameTracer
    RenameTracer = get_rename_tracer(Tracer)


def get_graph(
    model: nn.Module,
    tracer: Optional[Type[Tracer]] = None,
    concrete_args: Optional[Dict[str, Any]] = None,
) -> Graph:
    """Returns fx graph with named nodes"""
    rename_tracer = get_rename_tracer(tracer)() if exists(tracer) else RenameTracer()  # type: ignore # noqa
    graph = rename_tracer.trace(model, concrete_args=concrete_args)
    return graph


def get_nodes_alias(
    module: nn.Module,
    tracer: Optional[Type[Tracer]] = None,
    concrete_args: Optional[Dict[str, Any]] = None,
) -> List[str]:
    """Get node names from module"""
    graph = get_graph(module, tracer, concrete_args)
    return [
        node.alias
        for node in graph.nodes
        if node.op != "get_attr" and node.op != "output"
    ]


class Extract(GraphModule):
    def __init__(
        self,
        model: nn.Module,
        node_in: Optional[Union[str, Sequence[str], Dict[str, str]]] = None,
        node_out: Optional[Union[str, Sequence[str], Dict[str, str]]] = None,
        tracer: Optional[Type[Tracer]] = None,
        concrete_args: Optional[Dict[str, Any]] = None,
        keep_output: bool = None,
        share_modules: bool = False,
    ):
        # Deepcopy model if modules are not shared
        model = copy.deepcopy(model) if not share_modules else model
        # Build graph module
        graph = get_graph(model, tracer, concrete_args)
        super().__init__(model, graph)
        # If node_out provided we assume that current must out must not be kept
        keep_old: bool = keep_output if exists(keep_output) else not exists(node_out)  # type: ignore # noqa
        # Update graph
        self.update_graph_outputs(node_out, keep_old)
        self.update_graph_inputs(node_in)
        self.compute_summary()

    def update_graph_inputs(
        self, node_alias: Optional[Union[str, Sequence[str], Dict[str, str]]]
    ):
        if not exists(node_alias):
            return

        graph = self.graph
        aliases = to_list(node_alias)

        # Replace nodes with input nodes
        for node in graph.nodes:
            for alias in aliases:
                if getattr(node, "alias", None) == alias:
                    with graph.inserting_before(node):
                        new_name = (
                            node_alias[alias]
                            if isinstance(node_alias, dict)
                            else node.name
                        )
                        new_node = graph.placeholder(node.name)
                        # Name to set after cleaning
                        new_node.rename = new_name
                        node.replace_all_uses_with(new_node)
                    graph.erase_node(node)
        self.update()

        # Rename new input nodes and remove unused placeholders
        for node in graph.nodes:
            if node.op == "placeholder":
                if not node.users:
                    graph.erase_node(node)
                if getattr(node, "rename", None):
                    node.name = node.rename
                    node.target = node.rename
                    node.rename = None
        self.update()

    def update_graph_outputs(
        self,
        node_alias: Optional[Union[str, Sequence[str], Dict[str, str]]],
        keep_old: bool,
    ):
        if not exists(node_alias):
            return

        graph = self.graph
        aliases = to_list(node_alias)
        new_out_nodes = []
        old_out_node = None

        for node in graph.nodes:
            # Save old outputs args if keep and remove node
            if node.op == "output":
                old_out_node = node.args[0]
                graph.erase_node(node)
            # Save node as output if output
            for alias in aliases:
                if getattr(node, "alias", None) == alias:
                    new_out_nodes.append(node)

        # Format output according to node_out structure
        out_nodes = None
        if isinstance(node_alias, dict):
            out_nodes = {}
            for node in new_out_nodes:
                name = node_alias.get(node.alias, node.alias)
                out_nodes[name] = node
        elif isinstance(node_alias, tuple):
            out_nodes = tuple(new_out_nodes)  # type: ignore
        elif isinstance(node_alias, str):
            out_nodes = new_out_nodes[0]
        else:
            out_nodes = new_out_nodes  # type: ignore

        # Update graph output node
        node = graph.output((old_out_node, out_nodes) if keep_old else out_nodes)
        self.update()

    def compute_summary(self):
        summary = {"input": (), "output": None}
        for node in self.graph.nodes:
            if node.op == "placeholder":
                summary["input"] += (node.name,)
            if node.op == "output":
                summary["output"] = node.args[0]
        self.summary = summary

    def update(self):
        self.graph.lint()
        self.graph.eliminate_dead_code()
        self.recompile()
