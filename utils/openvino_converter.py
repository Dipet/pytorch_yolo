from models.yolov3_tiny import YOLOv3Tiny
from tensorboardX import SummaryWriter

import torch
from torch import jit
from torch import nn

from tensorboardX.pytorch_graph import NodePyIO, GraphPy, NodePyOP

import re

from collections import OrderedDict

from models.yolo_layer import YOLOLayer, Concat, Upsample

import numpy as np

from xml.dom import minidom
from xml.etree import ElementTree as ET

import os


def int_to_tuple(val, n=2):
    if not isinstance(val, int):
        return val

    return (val, ) * n


def repeat_tuple(val, n):
    if isinstance(val, int):
        return int_to_tuple(val, n)

    if len(val) == n:
        return val

    repeat = n / len(val) + 1
    val = val * repeat
    return val[:n]


class OpenVINOLayer:
    type = 'Base'

    def __init__(self, id, name, precision,
                 inputs=None, out_size=None, module=None):
        self.id = id
        self.name = name
        self.precision = precision
        self.inputs = inputs
        self.out_size = out_size[0] if out_size else out_size
        self.module = module
        self.blobs = {}
        self.data = {}

        i = 0
        self._input_ports = {}
        if inputs:
            for name, item in inputs.items():
                self._input_ports[name] = i
                i += 1

        self.output_port = i

    def input_port(self, name):
        return self._input_ports[name]

    def get_xml(self, blob_buff: bytearray):
        element = ET.Element('layer',
                             attrib={'id': str(self.id),
                                     'name': self.name,
                                     'precision': self.precision,
                                     'type': self.type,})
        if self.data:
            data = ET.Element('data', attrib={str(i): str(j) for i, j in self.data.items()})
            element.append(data)

        if self.inputs:
            input = ET.Element('input')

            for i, (_, size) in enumerate(self.inputs.items()):
                port = ET.Element('port', attrib={'id': str(i)})

                for d in size:
                    dim = ET.Element('dim')
                    dim.text = str(d)
                    port.append(dim)
                input.append(port)

            element.append(input)

        output = ET.Element('output')
        port = ET.Element('port', attrib={'id': str(self.output_port)})
        for d in self.out_size:
            dim = ET.Element('dim')
            dim.text = str(d)
            port.append(dim)
        output.append(port)
        element.append(output)

        dtype = np.float16 if self.precision == 'FP16' else np.float32
        if self.blobs:
            blobs = ET.Element('blobs')
            for key, array in self.blobs.items():
                offset = len(blob_buff)
                blob_buff += array.astype(dtype).tobytes()
                size = len(blob_buff) - offset
                blob = ET.Element(str(key), attrib={'offset': str(offset),
                                                    'size': str(size)})
                blobs.append(blob)
            element.append(blobs)

        return element


class OpenVINOInput(OpenVINOLayer):
    type = 'Input'


class OpenVINOConv2D(OpenVINOLayer):
    type = 'Convolution'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.data['auto_pad'] = 'same_upper'
        self.data['dilations'] = ','.join([str(i) for i in int_to_tuple(self.module.dilation)])
        self.data['kernel'] = ','.join([str(i) for i in  int_to_tuple(self.module.kernel_size)])
        self.data['output'] = str(self.module.out_channels)
        self.data['group'] = str(self.module.groups)
        self.data['pads_begin'] = '1,1'
        self.data['pads_end'] = '1,1'
        self.data['strides'] = ','.join([str(i) for i in int_to_tuple(self.module.stride)])

        state = self.module.state_dict()
        self.blobs['weights'] = state['weight'].detach().cpu().numpy()
        self.blobs['biases'] = state['bias'].detach().cpu().numpy()


class OpenVINOReLU(OpenVINOLayer):
    type = 'ReLU'


class OpenVINOMaxPool(OpenVINOLayer):
    type = 'Pooling'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if self.module.dilation == 2 and self.module.padding == 1:
            self.data['auto_pad'] = 'same_upper'
            self.data['pads_begin'] = '0,0'
            self.data['pads_end'] = '1,1'
        else:
            self.data['auto_pad'] = 'valid'
            self.data['pads_begin'] = '0,0'
            self.data['pads_end'] = '0,0'
        self.data['exclude-pad'] = 'true'
        self.data['kernel'] = ','.join([str(i) for i in int_to_tuple(self.module.kernel_size)])
        self.data['pool-method'] = 'max'
        self.data['strides'] = ','.join([str(i) for i in int_to_tuple(self.module.stride)])

class OpenVINOResample(OpenVINOLayer):
    type = 'Resample'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.data['antialias'] = str(0)
        self.data['factor'] = str(self.module.scale_factor)
        self.data['type'] = 'caffe.ResampleParameter.NEAREST'


class OpenVINOConcat(OpenVINOLayer):
    type = 'Concat'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        inputs = np.array([i for _, i in self.inputs.items()], dtype=int)
        self.out_size = inputs[0]
        self.out_size[1] = inputs[:, 1].sum()

        self.data['axis'] = str(self.module.dim)


class OpenVINORegionYolo(OpenVINOLayer):
    type = 'RegionYolo'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        for key, item in self.inputs.items():
            self.out_size = item

        anchors = np.array(self.module.all_anchors).flatten()
        self.data['anchors'] = ','.join([str(int(i)) for i in anchors])
        self.data['axis'] = str(1)
        self.data['coords'] = str(4)
        self.data['do_softmax'] = str(0)
        self.data['end_axis'] = str(3)
        self.data['mask'] = '0,1,2'
        self.data['num'] = str(len(self.module.anchors))
        self.data['classes'] = str(self.module.n_classes)


class OpenVINOLeakyReLU(OpenVINOLayer):
    type = 'ReLU'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.data['negative_slope'] = str(0.1)


class OpenVINOZeroPad(OpenVINOLayer):
    type = 'Pad'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.data['pad_mode'] = 'constant'
        self.data['pad_value'] = str(0)
        pads = repeat_tuple(self.module.padding, 4)
        self.data['pads_begin'] = ','.join([str(i) for i in pads])
        self.data['pads_end'] = ','.join([str(i) for i in pads])


class OpenVINOGraph:
    map_layers = {
        nn.Conv2d: OpenVINOConv2D,
        nn.MaxPool2d: OpenVINOMaxPool,
        nn.ReLU: OpenVINOReLU,
        nn.LeakyReLU: OpenVINOLeakyReLU,
        nn.ZeroPad2d: OpenVINOZeroPad,
        Upsample: OpenVINOResample,
        Concat: OpenVINOConcat,
        YOLOLayer: OpenVINORegionYolo,
    }

    def __init__(self, layers, precision='FP16'):
        self.layers = layers

        self.openvino_layers = OrderedDict()
        self.map_ind_layer = {}
        self.precision = precision

        self.create_layers()

    def add_layer(self, name, cls, *args, **kwargs):
        i = len(self.openvino_layers)
        self.openvino_layers[name] = cls(i, name, self.precision,
                                         *args, **kwargs)

    def create_layers(self):
        for key, item in self.layers.items():
            for inp, size in item['inputs'].items():
                if inp == 'input':
                    self.add_layer(inp, OpenVINOInput, out_size=size,)

            inputs = {i: self.openvino_layers[i].out_size for i in item['inputs']}  # need for concat
            self.add_layer(key, self.map_layers[item['module'].__class__],
                           inputs=inputs,
                           out_size=item['out_size'],
                           module=item['module'])

        return self.openvino_layers

    def save(self, name, save_dir=''):
        xml_name = os.path.join(save_dir, name + '.xml')
        bin_name = os.path.join(save_dir, name + '.bin')

        net = ET.Element('net', attrib={'name': name,
                                        'version': '5',
                                        'batch': '1'})
        layers = ET.Element('layers')
        blob_buff = bytearray()
        edges = ET.Element('edges')
        for i, (name, item) in enumerate(self.openvino_layers.items()):
            layers.append(item.get_xml(blob_buff))

            if item.inputs:
                for j, (key, _) in enumerate(item.inputs.items()):
                    input_node = self.openvino_layers[key]
                    edge = ET.Element('edge', attrib={'from-layer': str(input_node.id),
                                                      'from-port': str(input_node.output_port),
                                                      'to-layer': str(item.id),
                                                      'to-port': str(j)})
                    edges.append(edge)
        net.append(layers)
        net.append(edges)

        xml = ET.tostring(net)
        xml = minidom.parseString(xml)
        with open(xml_name, 'w') as file:
            file.write(xml.toprettyxml())

        with open(bin_name, 'wb') as file:
            file.write(blob_buff)


class MyGraph(GraphPy):
    def __init__(self):
        super().__init__()

        self.scopes = {}
        self.scopes_io = OrderedDict()

    @staticmethod
    def get_scope_name(name, input_node=False):
        name = name.split('/')

        for i in range(len(name)):
            val = name[i]
            if re.search(r'\[.+\]', val):
                val = re.sub(r'.*\[(.+)\].*', r'\1', val)
            name[i] = val

        if input_node:
            return '.'.join(name[:-1])
        return '.'.join(name)

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

                if name == scope or scope == '':
                    continue

                data = {
                    'input_size': node.inputstensor_size,
                    'output_size': node.outputstensor_size,
                    'node': node,
                    'input_id': input_id,
                    'input_name': name,
                }
                if scope in self.scopes_io:
                    self.scopes_io[scope].append(data)
                else:
                    self.scopes_io[scope] = [data]

    def get_attr_by_scope(self, obj, attr):
        attr = attr.split('.')

        res_obj = obj
        for item_name in attr:
            if item_name == obj.__class__.__name__:
                continue

            res_obj = getattr(res_obj, item_name, None)

        return res_obj

    def create_openvino_graph(self, model):
        for name, item in self.scopes_io.items():
            group_by_name = {'inputs': OrderedDict(),
                             'out_size': [],
                             'module': None,
                             'node': None,}
            group_by_name['module'] = self.get_attr_by_scope(model, name)
            for item_info in item:
                inp_size = [i for i in item_info['input_size'] if i is not None]
                out_size = [i for i in item_info['output_size'] if i is not None]

                group_by_name['out_size'] += out_size
                group_by_name['node'] = item_info['node']

                inp_name = item_info['input_name']
                if inp_name not in group_by_name['inputs'] and inp_size:
                    group_by_name['inputs'][inp_name] = inp_size

            self.scopes_io[name] = group_by_name

        if model.__class__.__name__ in self.scopes_io:
            self.scopes_io.pop(model.__class__.__name__)

        return OpenVINOGraph(self.scopes_io)


def parse(model, args, omit_useless_nodes=True):
    with torch.onnx.set_training(model, False):
        trace = jit.trace(model, args)
        graph = trace.graph

    n_inputs = len(args)  # not sure...

    nodes_py = MyGraph()
    for i, node in enumerate(graph.inputs()):
        if omit_useless_nodes:
            if len(
                    node.uses()) == 0:  # number of user of the node (= number of outputs/ fanout)
                continue

        if i < n_inputs:
            nodes_py.append(NodePyIO(node, 'input'))
        else:
            nodes_py.append(NodePyIO(node))  # parameter

    for node in graph.nodes():
        nodes_py.append(NodePyOP(node))

    for node in graph.outputs():  # must place last.
        NodePyIO(node, 'output')
    nodes_py.find_common_root()
    nodes_py.populate_namespace_from_OP_to_IO()

    nodes_py.parse_scopes()

    openvino_graph = nodes_py.create_openvino_graph(model)

    return nodes_py, openvino_graph


if __name__ == '__main__':
    device = 'cpu'
    img_size = 320
    input_shape = (1, 3, img_size, img_size)
    from torchsummary import summary

    model = YOLOv3Tiny(n_class=4, onnx=False, in_shape=input_shape, kernels_divider=3).to(device).eval()
    summary(model, input_shape[1:], device=device)
    model.fuse()
    writer = SummaryWriter()
    dummy = torch.rand(input_shape).to(device)

    try:
        _, nodes = parse(model, dummy)
    except:
        _, nodes = parse(model, dummy)
    nodes.save('test')
