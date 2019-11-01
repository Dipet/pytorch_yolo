import torch

from torch import jit
from torch import onnx
from tensorboardX.pytorch_graph import NodePyIO, NodePyOP

from pytorch_yolo.openvino_converter.graph import ModelGraph


__all__ = ["OpenVINOConverter"]


class OpenVINOConverter:
    @staticmethod
    def parse(model, args, omit_useless_nodes=True):
        with onnx.set_training(model, False):
            trace = jit.trace(model, args)
            graph = trace.graph

        n_inputs = args.shape[0]  # not sure...

        model_graph = ModelGraph()
        for i, node in enumerate(graph.inputs()):
            if omit_useless_nodes:
                if len(node.uses()) == 0:  # number of user of the node (= number of outputs/ fanout)
                    continue

            if i < n_inputs:
                model_graph.append(NodePyIO(node, "input"))
            else:
                model_graph.append(NodePyIO(node))  # parameter

        for node in graph.nodes():
            model_graph.append(NodePyOP(node))

        for node in graph.outputs():  # must place last.
            NodePyIO(node, "output")
        model_graph.find_common_root()
        model_graph.populate_namespace_from_OP_to_IO()

        model_graph.parse_scopes()

        openvino_graph = model_graph.create_openvino_graph(model)

        return model_graph, openvino_graph

    @staticmethod
    def save(model, args, name, save_dir=""):
        try:
            _, nodes = OpenVINOConverter.parse(model, args)
        except:
            _, nodes = OpenVINOConverter.parse(model, args)
        nodes.save(name, save_dir)
