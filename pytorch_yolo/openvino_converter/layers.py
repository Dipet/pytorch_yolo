import numpy as np

from xml.etree import ElementTree as ET


def _int_to_tuple(val, n=2):
    if not isinstance(val, int):
        return val

    return (val,) * n


def _repeat_tuple(val, n):
    if isinstance(val, int):
        return _int_to_tuple(val, n)

    if len(val) == n:
        return val

    repeat = n / len(val) + 1
    val = val * repeat
    return val[:n]


class OpenVINOLayer:
    type = "Base"

    def __init__(self, id, name, precision, inputs=None, out_size=None, module=None):
        self.id = id
        self.name = name
        self.precision = precision
        self.inputs = inputs
        self.out_size = out_size[0] if out_size else out_size
        self.module = module
        self.blobs = {}
        self.data = {}

        self._input_ports = {}
        self.output_port = 0
        if inputs:
            self.update_inputs(inputs)

    def input_port(self, name):
        return self._input_ports[name]

    def update_inputs(self, inputs):
        if not isinstance(inputs, dict):
            inputs = {inputs: None, **self._input_ports}

        i = 0
        for name, item in inputs.items():
            self._input_ports[name] = i
            i += 1
        self.output_port = i

    def get_xml(self, blob_buff: bytearray):
        element = ET.Element(
            "layer", attrib={"id": str(self.id), "name": self.name, "precision": self.precision, "type": self.type}
        )
        if self.data:
            data = ET.Element("data", attrib={str(i): str(j) for i, j in self.data.items()})
            element.append(data)

        if self.inputs:
            input = ET.Element("input")

            for i, (_, size) in enumerate(self.inputs.items()):
                port = ET.Element("port", attrib={"id": str(i)})

                for d in size:
                    dim = ET.Element("dim")
                    dim.text = str(d)
                    port.append(dim)
                input.append(port)

            element.append(input)

        output = ET.Element("output")
        port = ET.Element("port", attrib={"id": str(self.output_port)})
        for d in self.out_size:
            dim = ET.Element("dim")
            dim.text = str(d)
            port.append(dim)
        output.append(port)
        element.append(output)

        dtype = np.float16 if self.precision == "FP16" else np.float32
        if self.blobs:
            blobs = ET.Element("blobs")
            for key, array in self.blobs.items():
                offset = len(blob_buff)
                blob_buff += array.astype(dtype).tobytes()
                size = len(blob_buff) - offset
                blob = ET.Element(str(key), attrib={"offset": str(offset), "size": str(size)})
                blobs.append(blob)
            element.append(blobs)

        return element


class OpenVINOInput(OpenVINOLayer):
    type = "Input"


class OpenVINOConv2D(OpenVINOLayer):
    type = "Convolution"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.data["auto_pad"] = "same_upper"
        self.data["dilations"] = ",".join([str(i) for i in _int_to_tuple(self.module.dilation)])
        self.data["kernel"] = ",".join([str(i) for i in _int_to_tuple(self.module.kernel_size)])
        self.data["output"] = str(self.module.out_channels)
        self.data["group"] = str(self.module.groups)
        self.data["pads_begin"] = "1,1"
        self.data["pads_end"] = "1,1"
        self.data["strides"] = ",".join([str(i) for i in _int_to_tuple(self.module.stride)])

        state = self.module.state_dict()
        self.blobs["weights"] = state["weight"].detach().cpu().numpy()
        self.blobs["biases"] = state["bias"].detach().cpu().numpy()


class OpenVINOReLU(OpenVINOLayer):
    type = "ReLU"


class OpenVINOMaxPool(OpenVINOLayer):
    type = "Pooling"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if self.module.dilation == 2 and self.module.padding == 1:
            self.data["auto_pad"] = "same_upper"
            self.data["pads_begin"] = "0,0"
            self.data["pads_end"] = "1,1"
        else:
            self.data["auto_pad"] = "valid"
            self.data["pads_begin"] = "0,0"
            self.data["pads_end"] = "0,0"
        self.data["exclude-pad"] = "true"
        self.data["kernel"] = ",".join([str(i) for i in _int_to_tuple(self.module.kernel_size)])
        self.data["pool-method"] = "max"
        self.data["strides"] = ",".join([str(i) for i in _int_to_tuple(self.module.stride)])


class OpenVINOResample(OpenVINOLayer):
    type = "Resample"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.data["antialias"] = str(0)
        self.data["factor"] = str(self.module.scale_factor)
        self.data["type"] = "caffe.ResampleParameter.NEAREST"


class OpenVINOConcat(OpenVINOLayer):
    type = "Concat"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        inputs = np.array([i for _, i in self.inputs.items()], dtype=int)
        self.out_size = inputs[0]
        self.out_size[1] = inputs[:, 1].sum()

        self.data["axis"] = str(self.module.dim)


class OpenVINORegionYolo(OpenVINOLayer):
    type = "RegionYolo"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        for key, item in self.inputs.items():
            self.out_size = item

        anchors = np.array(self.module.all_anchors).flatten()
        self.data["anchors"] = ",".join([str(int(i)) for i in anchors])
        self.data["axis"] = str(1)
        self.data["coords"] = str(4)
        self.data["do_softmax"] = str(0)
        self.data["end_axis"] = str(3)
        self.data["mask"] = "0,1,2"
        self.data["num"] = str(len(self.module.anchors))
        self.data["classes"] = str(self.module.n_classes)


class OpenVINOLeakyReLU(OpenVINOLayer):
    type = "ReLU"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.data["negative_slope"] = str(0.1)


class OpenVINOAdd(OpenVINOLayer):
    type = "Eltwise"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.data["operation"] = "sum"
