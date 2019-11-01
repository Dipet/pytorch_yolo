import os
import re

from .layers import *
from torch import nn
from collections import OrderedDict
from pytorch_yolo.models.yolo_layer import Upsample, Concat, YOLOLayer
from pytorch_yolo.models.yolov3_spp import Add
from tensorboardX.pytorch_graph import GraphPy

from xml.dom import minidom
from pytorch_yolo.models.yolo_base import MaxPool


class OpenVINOGraph:
    map_layers = {
        nn.Conv2d: OpenVINOConv2D,
        nn.MaxPool2d: OpenVINOMaxPool,
        nn.ReLU: OpenVINOReLU,
        nn.LeakyReLU: OpenVINOLeakyReLU,
        Upsample: OpenVINOResample,
        Concat: OpenVINOConcat,
        YOLOLayer: OpenVINORegionYolo,
        MaxPool: OpenVINOMaxPool,
        Add: OpenVINOAdd,
    }

    def __init__(self, layers, precision="FP16"):
        self.layers = layers

        self.openvino_layers = OrderedDict()
        self.map_ind_layer = {}
        self.precision = precision

        self.create_layers()

    def add_layer(self, name, cls, *args, **kwargs):
        i = len(self.openvino_layers)
        self.openvino_layers[name] = cls(i, name, self.precision, *args, **kwargs)

    def create_layers(self):
        input_is_set = False
        flag_input = False
        need_add = {}
        for key, item in self.layers.items():
            if not input_is_set:
                for inp, size in item["inputs"].items():
                    input_is_set = True
                    self.add_layer(inp, OpenVINOInput, out_size=size)

            inputs = {}
            for i in item["inputs"]:
                if not flag_input or i.find(".") != -1:
                    if i in self.openvino_layers:
                        inputs[i] = self.openvino_layers[i].out_size
                    else:
                        if i in need_add:
                            need_add[i].append(key)
                        else:
                            need_add[i] = [key]

            if input_is_set:
                flag_input = True
            self.add_layer(
                key,
                self.map_layers[item["module"].__class__],
                inputs=inputs,
                out_size=item["out_size"],
                module=item["module"],
            )

            if key in need_add:
                items = need_add.pop(key)

                for item in items:
                    self.openvino_layers[item].update_inputs(key)

        return self.openvino_layers

    def save(self, name, save_dir=""):
        xml_name = os.path.join(save_dir, name + ".xml")
        bin_name = os.path.join(save_dir, name + ".bin")

        net = ET.Element("net", attrib={"name": name, "version": "5", "batch": "1"})
        layers = ET.Element("layers")
        blob_buff = bytearray()
        edges = ET.Element("edges")
        for i, (name, item) in enumerate(self.openvino_layers.items()):
            layers.append(item.get_xml(blob_buff))

            if item.inputs:
                for j, (key, _) in enumerate(item.inputs.items()):
                    input_node = self.openvino_layers[key]
                    edge = ET.Element(
                        "edge",
                        attrib={
                            "from-layer": str(input_node.id),
                            "from-port": str(input_node.output_port),
                            "to-layer": str(item.id),
                            "to-port": str(j),
                        },
                    )
                    edges.append(edge)
        net.append(layers)
        net.append(edges)

        xml = ET.tostring(net)
        xml = minidom.parseString(xml)
        with open(xml_name, "w") as file:
            file.write(xml.toprettyxml())

        with open(bin_name, "wb") as file:
            file.write(blob_buff)


class ModelGraph(GraphPy):
    def __init__(self):
        super().__init__()

        self.scopes = {}
        self.scopes_io = OrderedDict()

    @staticmethod
    def get_scope_name(name, input_node=False):
        name = name.split("/")

        for i in range(len(name)):
            val = name[i]
            if re.search(r"\[.+\]", val):
                val = re.sub(r".*\[(.+)\].*", r"\1", val)
            name[i] = val

        if input_node:
            return ".".join(name[:-1])
        return ".".join(name)

    def parse_scopes(self):
        for node in self.nodes_op:
            scope = self.get_scope_name(node.scopeName)

            if scope in self.scopes:
                self.scopes[scope].append(node)
            else:
                self.scopes[scope] = [node]

            for input_id in node.inputs:
                name = self.unique_name_to_scoped_name[input_id]
                name = self.get_scope_name(name, True)

                if name == scope or scope == "":
                    continue

                data = {
                    "input_size": node.inputstensor_size,
                    "output_size": node.outputstensor_size,
                    "node": node,
                    "input_id": input_id,
                    "input_name": name,
                }
                if scope in self.scopes_io:
                    self.scopes_io[scope].append(data)
                else:
                    self.scopes_io[scope] = [data]

    def get_attr_by_scope(self, obj, attr):
        attr = attr.split(".")

        res_obj = obj
        for item_name in attr:
            if item_name == obj.__class__.__name__:
                continue

            _obj = getattr(res_obj, item_name, None)
            if _obj is None:
                continue

            res_obj = _obj

        return res_obj

    def create_openvino_graph(self, model):
        for name, item in self.scopes_io.items():
            group_by_name = {
                "inputs": OrderedDict(),
                "out_size": [],
                "module": self.get_attr_by_scope(model, name),
                "node": None,
            }
            for item_info in item:
                inp_size = [i for i in item_info["input_size"] if i is not None]
                out_size = [i for i in item_info["output_size"] if i is not None]

                group_by_name["out_size"] += out_size
                group_by_name["node"] = item_info["node"]

                inp_name = item_info["input_name"]
                if inp_name not in group_by_name["inputs"] and inp_size:
                    group_by_name["inputs"][inp_name] = inp_size

            self.scopes_io[name] = group_by_name

        if model.__class__.__name__ in self.scopes_io:
            self.scopes_io.pop(model.__class__.__name__)

        return OpenVINOGraph(self.scopes_io)
