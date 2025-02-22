## HW 2-1 Model Analysis Using Pytorch

### 2-1-1. Calculate the number of model parameters：

#### Code
```python=
import torchvision.models as models

model = models.googlenet(pretrained=True)
# p.numel() 取得 p 所有 parameter 維度相乘結果
total_params = sum(p.numel() for p in model.parameters())
print("Total number of parameters: ", total_params)
```

#### Execution Result
![](https://course.playlab.tw/md/uploads/7fb0cc47-2123-4d89-83fd-42c6cec0a9c0.png)

### 2-1-2. Calculate memory requirements for storing the model weights.
#### Code
```python=
import torchvision.models as models

model = models.googlenet(pretrained=True)
# p.element_size() 取得 p 的 element 占多少 byte
param_size = sum(p.numel() * p.element_size() for p in model.parameters())
print("Total memory for parameters: ", param_size)
```

#### Execution Result
![](https://course.playlab.tw/md/uploads/2dd6a1af-2711-434c-99d8-28ec2b66d6a8.png)

### 2-1-3. Use Torchinfo to print model architecture including the number of parameters and the output activation size of each layer 
#### Code
```python=
import torchinfo
import torchvision.models as models

model = models.googlenet(pretrained=True)
# depth 取 4，googlenet 有第 4 層 layer
torchinfo.summary(model, (3, 224, 224), batch_dim=0, depth=4, col_names=("input_size", "output_size", "num_params"))
```

#### Execution Result
![](https://course.playlab.tw/md/uploads/42eb817a-c54f-4dee-8bb9-58df7656b040.png)

### 2-1-4. Calculate computation requirements
#### Code
```python=
import math
import torch.nn as nn
import torchvision.models as models

def calculate_output_shape(input_shape, layer):
    if isinstance(layer, (nn.Conv2d, nn.MaxPool2d)):
        kernel_size = (
            layer.kernel_size
            if isinstance(layer.kernel_size, tuple)
            else (layer.kernel_size, layer.kernel_size)
        )
        stride = (
            layer.stride
            if isinstance(layer.stride, tuple)
            else (layer.stride, layer.stride)
        )
        padding = (
            layer.padding
            if isinstance(layer.padding, tuple)
            else (layer.padding, layer.padding)
        )
        dilation = (
            layer.dilation
            if isinstance(layer.dilation, tuple)
            else (layer.dilation, layer.dilation)
        )
        # ceil_mode 決定要用天花板/地板除
        if isinstance(layer, nn.MaxPool2d) and layer.ceil_mode:
            output_height = math.ceil((
                input_shape[1] + 2 * padding[0] - dilation[0] * (kernel_size[0] - 1) - 1
            ) / stride[0] + 1)
            output_width = math.ceil((
                input_shape[2] + 2 * padding[1] - dilation[1] * (kernel_size[1] - 1) - 1
            ) / stride[1] + 1)
        else:
            output_height = (
                input_shape[1] + 2 * padding[0] - dilation[0] * (kernel_size[0] - 1) - 1
            ) // stride[0] + 1
            output_width = (
                input_shape[2] + 2 * padding[1] - dilation[1] * (kernel_size[1] - 1) - 1
            ) // stride[1] + 1
        return (
            layer.out_channels if hasattr(layer, "out_channels") else input_shape[0],
            output_height,
            output_width,
        )
    elif isinstance(layer, nn.Linear):
        return (layer.out_features)
    elif isinstance(layer, nn.AdaptiveAvgPool2d):
        # 處理 AdaptiveAvgPool2d shape
        output_size = (
            layer.output_size
            if isinstance(layer.output_size, tuple)
            else (layer.output_size, layer.output_size)
        )
        # output_size 可以是 None，如果是 None 維度就取 input 維度
        if output_size[0] is None and output_size[1] is None:
            return input_shape
        elif output_size[0] is None:
            return input_shape[:len(input_shape) - 1] + output_size[1]
        elif output_size[1] is None:
            return input_shape[:len(input_shape) - 2] + output_size[0] + input_shape[-1]
        else:
            return input_shape[:len(input_shape) - 2] + output_size
    else:
        return input_shape

def calculate_macs(layer, output_shape):
    if isinstance(layer, nn.Conv2d):
        kernel_ops = (
            layer.kernel_size[0]
            * layer.kernel_size[1]
            * (layer.in_channels / layer.groups)
        )
        output_elements = output_shape[1] * output_shape[2]
        macs = int(kernel_ops * output_elements * layer.out_channels)
        return macs
    elif isinstance(layer, nn.Linear):
        macs = int(layer.in_features * layer.out_features)
        # 計算 linear layer 加 bias 的 MACs
        if layer.bias is not None:
            macs += layer.out_features
        return macs
    elif isinstance(layer, nn.BatchNorm2d):
        # BatchNorm2d 的 MACs 等於 input channel * 2
        return 2 * output_shape[0]
    else:
        return 0
    
def not_inception_count(layer, input_shape, total_macs, check_inception):
    for sub_layer_name, sub_layer in layer.named_children():
        if check_inception and "inception" in sub_layer_name:
            input_shape, total_macs = inception_count(sub_layer, input_shape, total_macs)
        else:
            if isinstance(sub_layer, (nn.Conv2d, nn.MaxPool2d, nn.Linear, nn.AdaptiveAvgPool2d, nn.BatchNorm2d)):
                input_shape = calculate_output_shape(input_shape, sub_layer)
                macs = calculate_macs(sub_layer, input_shape)
                total_macs += macs
            else:
                input_shape, total_macs = not_inception_count(sub_layer, input_shape, total_macs, True)
            
    return input_shape, total_macs

# inception layer 的 output channel 等於 sublayer output channel 加總
def inception_count(layer, input_shape, total_macs):
    output_shapes = []
    for sub_layer in layer.children():
        output_shape, total_macs = not_inception_count(sub_layer, input_shape, total_macs, False)
        output_shapes.append(output_shape)
    return tuple(item if i != 0 else sum(item[0] for item in output_shapes) for i, item in enumerate(output_shapes[0])), total_macs

model = models.googlenet(pretrained=True)
total_macs = 0
input_shape = (3, 224, 224)
output_shape, total_macs = not_inception_count(model, input_shape, total_macs, True)

print(f"Total MACs: {total_macs}")
```

#### Execution Result
![](https://course.playlab.tw/md/uploads/7d032680-cd96-4ca1-bde6-99a5a03cbc88.png)

### 2-1-5. Use forward hooks to extract the output activations of  the Conv2d layers.
#### Code
```python=
import torch
import torch.nn as nn
import torchvision.models as models

def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook

model = models.googlenet(pretrained=True)
activation = {}

for layer_name, layer in model.named_modules():
    if isinstance(layer, nn.Conv2d):
        # 註冊 Conv2d layer
        layer.register_forward_hook(get_activation(layer_name))

data = torch.randn(1, 3, 224, 224)
output = model(data)

for layer in activation:
    print(f"Activation from layer {layer}: {activation[layer]}")
```

#### Execution Result
![](https://course.playlab.tw/md/uploads/16c5a33a-f153-45ca-8ede-9989f27d32bd.png)

## HW 2-2 Add more statistics to analyze the an ONNX model

### 2-2-1. model characteristics
#### Code
```python=
import onnx

inputs = dict()
initializers = dict()
value_infos = dict()
outputs = dict()

# 從 4 個 dictionary 中取值
def get_value_by_key(key):
    if key in inputs.keys():
        return inputs[key]
    elif key in initializers.keys():
        return initializers[key]
    elif key in value_infos.keys():
        return value_infos[key]
    elif key in outputs.keys():
        return outputs[key]
    else:
        return key

def parse_shape(shape):
    dims = []
    for dim in shape.dim:
        if dim.HasField('dim_param'):
            dims.append(dim.dim_param)
        elif dim.HasField('dim_value'):
            dims.append(dim.dim_value)
    return dims

onnx_model = onnx.load('./mobilenetv2-10.onnx')
onnx_model = onnx.shape_inference.infer_shapes(onnx_model)

# 依序取出 model 的 input, initializer, value_info, output 建立 dictionary
for input in onnx_model.graph.input:
    parsed_input = dict()
    parsed_input["dim"] = tuple(parse_shape(input.type.tensor_type.shape))
    parsed_input["elem_type"] = input.type.tensor_type.elem_type
    inputs[input.name] = parsed_input

for init in onnx_model.graph.initializer:
    parsed_init = dict()
    parsed_init["dim"] = tuple(init.dims)
    parsed_init["elem_type"] = init.data_type
    initializers[init.name] = parsed_init

for value_info in onnx_model.graph.value_info:
    parsed_value_info = dict()
    parsed_value_info["dim"] = tuple(parse_shape(value_info.type.tensor_type.shape))
    parsed_value_info["elem_type"] = value_info.type.tensor_type.elem_type
    value_infos[value_info.name] = parsed_value_info

for output in onnx_model.graph.output:
    parsed_output = dict()
    parsed_output["dim"] = tuple(parse_shape(output.type.tensor_type.shape))
    parsed_output["elem_type"] = output.type.tensor_type.elem_type
    outputs[output.name] = parsed_output

# 2-2-1-1
ops = dict()
for i in onnx_model.graph.node:
    if i.op_type not in ops.keys():
        ops[i.op_type] = 1
    else:
        ops[i.op_type] += 1
print(f"Operator types: {ops}")

# 2-2-1-2
operator_attributes = dict()
for i in onnx_model.graph.node:
    attrs = dict()
    # Constant, Gather, Unsqueeze, Concat, Gemm 沒有 input
    if not any(op in i.name for op in ["Constant", "Gather", "Unsqueeze", "Concat", "Gemm"]):
        attrs["input_dim"] = get_value_by_key(i.input[0])["dim"]
    # Shape, Constant, Gather, Unsqueeze, Concat, Reshape 沒有 output
    if not any(op in i.name for op in ["Shape", "Constant", "Gather", "Unsqueeze", "Concat", "Reshape"]):
        attrs["output_dim"] = get_value_by_key(i.output[0])["dim"]
    # 取剩下的 attribute
    for attr in i.attribute:
        attr_str = str(attr).strip()
        attrs[attr.name] = ", ".join(attr_str.split("\n")[1:])
    operator_attributes[i.name] = attrs
print("Operator attributes")
for key,value in operator_attributes.items():
    print(key)
    print(value)
```

#### Execution Result
![](https://course.playlab.tw/md/uploads/8e3b215d-51fd-4b36-9eb8-2e333521c23d.png)

### 2-2-2. Data bandwidth requirement 
#### Code
```python=
import onnx

inputs = dict()
initializers = dict()
value_infos = dict()
outputs = dict()

def mutiply_tuple(input_value):
    if len(input_value["dim"]) == 0:
        return 0
    
    # 將 batch_size 以 1 替換
    dims = (1 if elem == 'batch_size' else elem for elem in input_value["dim"])
    result = 1

    for element in dims:
        result *= element

    # 判斷一個 element 多少 byte
    match input_value["elem_type"]:
        case 1:
            result *= 4
        case 2:
            result *= 1
        case 3:
            result *= 1
        case 4:
            result *= 2
        case 5:
            result *= 2
        case 6:
            result *= 4
        case 7:
            result *= 8
        case 9:
            result *= 1
        case 10:
            result *= 2
        case 11:
            result *= 8
        case 12:
            result *= 4
        case 13:
            result *= 8
        case 14:
            result *= 8
        case 15:
            result *= 16
        case _:
            print("Undefined data type")

    return result

def get_value_by_key(key):
    if key in inputs.keys():
        return inputs[key]
    elif key in initializers.keys():
        return initializers[key]
    elif key in value_infos.keys():
        return value_infos[key]
    elif key in outputs.keys():
        return outputs[key]
    else:
        return key

def parse_shape(shape):
    dims = []
    for dim in shape.dim:
        if dim.HasField('dim_param'):
            dims.append(dim.dim_param)
        elif dim.HasField('dim_value'):
            dims.append(dim.dim_value)
    return dims

onnx_model = onnx.load('./mobilenetv2-10.onnx')
onnx_model = onnx.shape_inference.infer_shapes(onnx_model)

for input in onnx_model.graph.input:
    parsed_input = dict()
    parsed_input["dim"] = tuple(parse_shape(input.type.tensor_type.shape))
    parsed_input["elem_type"] = input.type.tensor_type.elem_type
    inputs[input.name] = parsed_input

for init in onnx_model.graph.initializer:
    parsed_init = dict()
    parsed_init["dim"] = tuple(init.dims)
    parsed_init["elem_type"] = init.data_type
    initializers[init.name] = parsed_init

for value_info in onnx_model.graph.value_info:
    parsed_value_info = dict()
    parsed_value_info["dim"] = tuple(parse_shape(value_info.type.tensor_type.shape))
    parsed_value_info["elem_type"] = value_info.type.tensor_type.elem_type
    value_infos[value_info.name] = parsed_value_info

for output in onnx_model.graph.output:
    parsed_output = dict()
    parsed_output["dim"] = tuple(parse_shape(output.type.tensor_type.shape))
    parsed_output["elem_type"] = output.type.tensor_type.elem_type
    outputs[output.name] = parsed_output

input_bandwith = 0
output_bandwith = 0

for i in onnx_model.graph.node:
    for input_name in i.input:
        input_bandwith += mutiply_tuple(get_value_by_key(input_name))
    for output_name in i.output:
        output_bandwith += mutiply_tuple(get_value_by_key(output_name))

total_bandwith = input_bandwith + output_bandwith

print(f"Data bandwidth requirement: {total_bandwith}")
```

#### Execution Result
![](https://course.playlab.tw/md/uploads/dfdc90ed-11f5-4269-be15-1b36e38a4d12.png)

### 2-2-3. activation memory storage requirement
#### Code
```python=
import onnx

inputs = dict()
initializers = dict()
value_infos = dict()
outputs = dict()

def mutiply_tuple(input_value):
    if len(input_value["dim"]) == 0:
        return 0
    
    dims = (1 if elem == 'batch_size' else elem for elem in input_value["dim"])
    result = 1

    for element in dims:
        result *= element

    match input_value["elem_type"]:
        case 1:
            result *= 4
        case 2:
            result *= 1
        case 3:
            result *= 1
        case 4:
            result *= 2
        case 5:
            result *= 2
        case 6:
            result *= 4
        case 7:
            result *= 8
        case 9:
            result *= 1
        case 10:
            result *= 2
        case 11:
            result *= 8
        case 12:
            result *= 4
        case 13:
            result *= 8
        case 14:
            result *= 8
        case 15:
            result *= 16
        case _:
            print("Undefined data type")

    return result

def get_value_by_key(key):
    if key in inputs.keys():
        return inputs[key]
    elif key in initializers.keys():
        return initializers[key]
    elif key in value_infos.keys():
        return value_infos[key]
    elif key in outputs.keys():
        return outputs[key]
    else:
        return key

def parse_shape(shape):
    dims = []
    for dim in shape.dim:
        if dim.HasField('dim_param'):
            dims.append(dim.dim_param)
        elif dim.HasField('dim_value'):
            dims.append(dim.dim_value)
    return dims

onnx_model = onnx.load('./mobilenetv2-10.onnx')
onnx_model = onnx.shape_inference.infer_shapes(onnx_model)

for input in onnx_model.graph.input:
    parsed_input = dict()
    parsed_input["dim"] = tuple(parse_shape(input.type.tensor_type.shape))
    parsed_input["elem_type"] = input.type.tensor_type.elem_type
    inputs[input.name] = parsed_input

for init in onnx_model.graph.initializer:
    parsed_init = dict()
    parsed_init["dim"] = tuple(init.dims)
    parsed_init["elem_type"] = init.data_type
    initializers[init.name] = parsed_init

for value_info in onnx_model.graph.value_info:
    parsed_value_info = dict()
    parsed_value_info["dim"] = tuple(parse_shape(value_info.type.tensor_type.shape))
    parsed_value_info["elem_type"] = value_info.type.tensor_type.elem_type
    value_infos[value_info.name] = parsed_value_info

for output in onnx_model.graph.output:
    parsed_output = dict()
    parsed_output["dim"] = tuple(parse_shape(output.type.tensor_type.shape))
    parsed_output["elem_type"] = output.type.tensor_type.elem_type
    outputs[output.name] = parsed_output

max_activation = 0

# 假設不需要的 output activation 會被清除，所需用來保存 output activation 的最小 storage requirement
for i in onnx_model.graph.node:
    # 從 Netron 結果來看 Add 跟 Reshape 需要兩個 input，因此不能只保留單一 layer 的 output activation
    if "Add" in i.name or "Reshape" in i.name:
        sum_activation = 0
        for input_name in i.input:
            sum_activation += mutiply_tuple(get_value_by_key(input_name))
        max_activation = max(max_activation, sum_activation)
    # 計算單一 layer 最大 output activation
    for output_name in i.output:
        max_activation = max(max_activation, mutiply_tuple(get_value_by_key(output_name)))

print(f"Activation memory storage requirement: {max_activation}")
```

#### Execution Result
![](https://course.playlab.tw/md/uploads/4fd1635f-eca8-4578-934f-bd03defc5793.png)

## HW 2-3 Build tool scripts to manipulate an ONNX model graph

### 2-3-1. create a subgraph (1) that consist of a single Linear layer of size MxKxN

#### Code
```python=
import torch
import torch.nn as nn

class Subgraph1(nn.Module):
    def __init__(self):
        super(Subgraph1, self).__init__()
    
    def forward(self, input1, input2):
        return torch.matmul(input1, input2)

input1 = torch.randn(128, 128)
input2 = torch.randn(128, 128)
subgraph1 = Subgraph1()
torch.onnx.export(subgraph1, (input1, input2), 'subgraph1.onnx', input_names=['A', 'B'], output_names=['C'])
```

#### Visualize the subgraph (1)
![](https://course.playlab.tw/md/uploads/89388a5e-337f-4bd1-bbc6-7c819beb3140.png)

### 2-3-2. create a subgraph (2) as shown in the above diagram for the subgraph (1)  

#### Code
```python=
import torch
import torch.nn as nn

class Subgraph2(nn.Module):
    def __init__(self):
        super(Subgraph2, self).__init__()
    
    # 先從 row split，再 split col，給 input1 用
    def _split_64_row_first(self, input_tensor):
        split_rows = torch.split(input_tensor, 64, dim=0)
        split_tensors = [torch.split(t, 64, dim=1) for t in split_rows]
        return split_tensors

    # 先從 col split，再 split row，給 input2 用
    def _split_64_col_first(self, input_tensor):
        split_cols = torch.split(input_tensor, 64, dim=1)
        split_tensors = [torch.split(t, 64, dim=0) for t in split_cols]
        return split_tensors

    # 依照矩陣乘法規則將小矩陣相乘後相加
    def _mul_sub_tensors(self, input_tensor_a, input_tensor_b):
        total_tensors = []
        for split_row_a in input_tensor_a:
            rowwise_tensors = []
            for split_col_b in input_tensor_b:
                mul_tensor = torch.zeros([split_row_a[0].shape[0], split_col_b[0].shape[1]])
                if len(split_row_a) != len(split_col_b):
                    raise ValueError
                for i in range(len(split_row_a)):
                    mul_tensor = torch.add(mul_tensor, torch.matmul(split_row_a[i], split_col_b[i]))
                rowwise_tensors.append(mul_tensor)
            total_tensors.append(rowwise_tensors)
        return total_tensors

    # concat _mul_sub_tensors 後的數個小矩陣得到最後的 output
    def _concat_tensors(self, tensors):
        colwise_concat_tensors = [torch.cat(tuple(row), dim=1) for row in tensors]
        return torch.cat(tuple(colwise_concat_tensors), dim=0)

    def forward(self, input1, input2):
        split_tensors_a = self._split_64_row_first(input1)
        split_tensors_b = self._split_64_col_first(input2)
        total_tensors = self._mul_sub_tensors(split_tensors_a, split_tensors_b)
        concat_tensor = self._concat_tensors(total_tensors)
        return concat_tensor

input1 = torch.randn(128, 128)
input2 = torch.randn(128, 128)
subgraph2 = Subgraph2()
torch.onnx.export(subgraph2, (input1, input2), 'subgraph2.onnx', input_names=['A', 'B'], output_names=['C'])
```

#### Visualize the subgraph (2)
![](https://course.playlab.tw/md/uploads/53bdee7a-ed93-45e0-b323-36397f06d015.png)


### 2-3-3. replace the `Linear` layers in the AlexNet with the equivalent subgraphs (2)
#### Code
```python=
import torch
import torch.nn as nn
import torchvision.models as models

def replace_linear_layer(model, new_module):
    for child_name, child_module in model.named_children():
        # 替換掉 linear layer 並取出 in_features, out_features, weight data, bias data
        if isinstance(child_module, nn.Linear):
            if child_module.bias.data is not None:
                setattr(model, child_name, new_module(child_module.in_features, child_module.out_features,
                                                      child_module.weight.data, bias_data=child_module.bias.data))
            else:
                setattr(model, child_name, new_module(child_module.in_features, child_module.out_features,
                                                      child_module.weight.data, bias=False))
        else:
            replace_linear_layer(child_module, new_module)

# 繼承 nn.Linear
class Split64Linear(nn.Linear):
    # 初始化 nn.Linear 後將 weight, bias copy 過去
    def __init__(self, in_features, out_features, weight_data, bias=True, bias_data=None):
        super(Split64Linear, self).__init__(in_features, out_features, bias)
        self.weight.data.copy_(weight_data)
        if bias_data is not None:
            self.bias.data.copy_(bias_data)
    
    # 先從 row split，再 split col，給 input 用
    def _split_64_row_first(self, input_tensor):
        split_rows = torch.split(input_tensor, 64, dim=0)
        split_tensors = [torch.split(t, 64, dim=1) for t in split_rows]
        return split_tensors

    # 先從 col split，再 split row，給 linear layer weight 用
    def _split_64_col_first(self, input_tensor):
        split_cols = torch.split(input_tensor, 64, dim=1)
        split_tensors = [torch.split(t, 64, dim=0) for t in split_cols]
        return split_tensors

    def _mul_sub_tensors(self, input_tensor_a, input_tensor_b):
        total_tensors = []
        for split_row_a in input_tensor_a:
            rowwise_tensors = []
            for split_col_b in input_tensor_b:
                mul_tensor = torch.zeros([split_row_a[0].shape[0], split_col_b[0].shape[1]])
                for i in range(len(split_row_a)):
                    mul_tensor = torch.add(mul_tensor, torch.matmul(split_row_a[i], split_col_b[i]))
                rowwise_tensors.append(mul_tensor)
            total_tensors.append(rowwise_tensors)
        return total_tensors

    def _concat_tensors(self, tensors):
        colwise_concat_tensors = [torch.cat(tuple(row), dim=1) for row in tensors]
        return torch.cat(tuple(colwise_concat_tensors), dim=0)

    def forward(self, input_tensor):
        split_tensors_a = self._split_64_row_first(input_tensor)
        # weight 要 transpose
        split_tensors_b = self._split_64_col_first(torch.transpose(self.weight, 0, 1))
        total_tensors = self._mul_sub_tensors(split_tensors_a, split_tensors_b)
        concat_tensor = self._concat_tensors(total_tensors)
        # 加上 bias
        if self.bias is not None:
            concat_tensor = torch.add(concat_tensor, self.bias)
        return concat_tensor

alexnet = models.alexnet(pretrained=True)
replace_linear_layer(alexnet, Split64Linear)
alexnet_input = torch.randn(1, 3, 224, 224)
torch.onnx.export(alexnet, alexnet_input, 'replaced_alexnet.onnx', input_names=['input'], output_names=['output'])
```

#### Visualize the transformed model graph
![](https://course.playlab.tw/md/uploads/0d6c5292-8321-40b2-8e17-d35e63ec870a.png)

### 2-3-4. Correctness Verification
#### Code
```python=
import copy
import torch
import torch.nn as nn
import torchvision.models as models

def replace_linear_layer(model, new_module):
    for child_name, child_module in model.named_children():
        if isinstance(child_module, nn.Linear):
            if child_module.bias.data is not None:
                setattr(model, child_name, new_module(child_module.in_features, child_module.out_features,
                                                      child_module.weight.data, bias_data=child_module.bias.data))
            else:
                setattr(model, child_name, new_module(child_module.in_features, child_module.out_features,
                                                      child_module.weight.data, bias=False))
        else:
            replace_linear_layer(child_module, new_module)

# 比較兩個 tensor
def compare_tensor(tensor1, tensor2):
    # 比較 shape
    if tensor1.shape != tensor2.shape:
        print(f"Output shape different")
    else:
        print(f"Same output shape {tensor1.shape}")
    
    # 找出兩個 tensor 中 element 最大差的範圍 (以 1e-x 表示)
    threshold = 1
    while True:
        compared_result = torch.sub(tensor1, tensor2) < threshold
        if torch.sum(~compared_result) > 0:
            print(f"{10 * threshold:.1e} > The maximum difference between tensor1 and tensor2 > {threshold:.1e}")
            break
        elif threshold < 1e-10:
            print(f"The maximum difference between tensor1 and tensor2 < {threshold:.1e}")
            break
        threshold /= 10
    

class CompareSplit64Linear(nn.Linear):
    def __init__(self, in_features, out_features, weight_data, bias=True, bias_data=None):
        super(CompareSplit64Linear, self).__init__(in_features, out_features, bias)
        self.weight.data.copy_(weight_data)
        if bias_data is not None:
            self.bias.data.copy_(bias_data)
    
    def _split_64_row_first(self, input_tensor):
        split_rows = torch.split(input_tensor, 64, dim=0)
        split_tensors = [torch.split(t, 64, dim=1) for t in split_rows]
        return split_tensors

    def _split_64_col_first(self, input_tensor):
        split_cols = torch.split(input_tensor, 64, dim=1)
        split_tensors = [torch.split(t, 64, dim=0) for t in split_cols]
        return split_tensors

    def _mul_sub_tensors(self, input_tensor_a, input_tensor_b):
        total_tensors = []
        for split_row_a in input_tensor_a:
            rowwise_tensors = []
            for split_col_b in input_tensor_b:
                mul_tensor = torch.zeros([split_row_a[0].shape[0], split_col_b[0].shape[1]])
                for i in range(len(split_row_a)):
                    mul_tensor = torch.add(mul_tensor, torch.matmul(split_row_a[i], split_col_b[i]))
                rowwise_tensors.append(mul_tensor)
            total_tensors.append(rowwise_tensors)
        return total_tensors

    def _concat_tensors(self, tensors):
        colwise_concat_tensors = [torch.cat(tuple(row), dim=1) for row in tensors]
        return torch.cat(tuple(colwise_concat_tensors), dim=0)

    def forward(self, input_tensor):
        split_tensors_a = self._split_64_row_first(input_tensor)
        split_tensors_b = self._split_64_col_first(torch.transpose(self.weight, 0, 1))
        total_tensors = self._mul_sub_tensors(split_tensors_a, split_tensors_b)
        concat_tensor = self._concat_tensors(total_tensors)
        if self.bias is not None:
            concat_tensor = torch.add(concat_tensor, self.bias)
        # 用 linear layer 原始的 forward 方法計算 output 並比較
        linear_tensor = super(CompareSplit64Linear, self).forward(input_tensor)
        print("Linear layer output compare")
        compare_tensor(concat_tensor, linear_tensor)
        return concat_tensor

alexnet_input = torch.randn(10, 3, 224, 224)
alexnet = models.alexnet(pretrained=True)
# copy alexnet，用 models.alexnet(pretrained=True) 再取得一次 alexnet 參數會不同
replaced_alexnet = copy.deepcopy(alexnet)
replace_linear_layer(replaced_alexnet, CompareSplit64Linear)
# evaluation 模式，避免 Dropout 等 layer 影響 inference 結果
alexnet.eval()
replaced_alexnet.eval()
original_output = alexnet(alexnet_input)
replaced_output = replaced_alexnet(alexnet_input)
# 比較兩個 model 的 inference 結果
print("Model output compare")
compare_tensor(original_output, replaced_output)
```

#### Execution Result
![](https://course.playlab.tw/md/uploads/7ab7a6b6-51e1-4a73-839f-c3fff054c060.png)

## HW 2-4 Using Pytorch C++ API to do model analysis on the transformed model graph

### 2-4-1. Calculate memory requirements for storing the model weights.

#### Code
```cpp=
#include <torch/script.h>

int main(int argc, const char* argv[]) {
    if (argc != 2) {
        std::cerr << "usage: ./hw2-4-1 <path-to-exported-script-module>\n";
        return -1;
    }

    torch::jit::script::Module module;
    try {
        module = torch::jit::load(argv[1]);
    }
    catch (const c10::Error& e) {
        std::cerr << "error loading the model\n";
        return -1;
    }

    int total_model_weight = 0;
    for (const auto& param : module.parameters()) {
        // 同 pytorch 用法
        total_model_weight += param.numel() * param.element_size();
    }
    std::cout << "Total model weights: " << total_model_weight << std::endl;
    
    return 0;
}
```

#### Execution Result
##### Replaced Alexnet
![](https://course.playlab.tw/md/uploads/3ef7e4c7-13da-453c-9fbe-ec861c918c3e.png)
##### Resnet18
![](https://course.playlab.tw/md/uploads/b97d2536-9d67-47b6-af27-4d09f709b5bf.png)

### 2-4-2. Calculate memory requirements for storing the activations

#### Code
```cpp
#include <torch/script.h>

void extract_input_output(torch::jit::NameModule module, torch::Tensor* input_tensor, int* t_a_m_r);

void getConstantParam(torch::jit::Node* input_node, std::string parme_type_string, torch::jit::Stack* torch_stack)
{
	if (input_node->hasAttributes())
    {
		auto parent_attr_names = input_node->attributeNames();
		auto parent_attr       = parent_attr_names[0];

		if (strcmp(parent_attr.toUnqualString(), "value") == 0)
        {
			auto parent_attr_kind = input_node->kindOfS("value");
			switch (parent_attr_kind)
            {
				case torch::jit::AttributeKind::i:
					if (parme_type_string.substr(0, 4) == "bool")
                    {
						torch_stack->push_back(input_node->i(parent_attr) == 1);
					}
                    else
                    {
						torch_stack->push_back(input_node->i(parent_attr));
					}
					break;
				case torch::jit::AttributeKind::f:
					torch_stack->push_back(input_node->f(parent_attr));
					break;
				case torch::jit::AttributeKind::t:
					torch_stack->push_back(input_node->t(parent_attr));
					break;
                // 處理 string
                case torch::jit::AttributeKind::s:
                    // 判斷 Device = prim::Constant[value="cpu"]()，push string 到 stack 在 operation 會錯
                    if (parme_type_string.substr(0, 6) == "Device" && input_node->s(parent_attr) == "cpu")
                    {
                        torch::Device device(torch::kCPU);
                        torch_stack->push_back(torch::IValue(device));
                    }
                    break;
			}
		}
	}
    else
    {
		torch_stack->push_back(torch::jit::IValue());
	}
}

void getListParam(torch::jit::Node* input_node, torch::jit::Stack* torch_stack, std::map<std::string, torch::IValue>* input_map)
{
	std::vector<int64_t>                             list_i;
	std::vector<float>                               list_f;
	std::vector<torch::Tensor>                       list_tensor;
    std::map<torch::jit::Value*, torch::jit::IValue> map_method_outputs;

	for (const auto& parent_in : input_node->inputs())
    {
        // 傳入紀錄計算結果的 map，並先從這個 map 開始搜尋
        if (input_map->find(parent_in->debugName()) != input_map->end())
        {
            if ((*input_map)[parent_in->debugName()].isInt())
            {
                list_i.push_back((*input_map)[parent_in->debugName()].toInt());
            }
            else if ((*input_map)[parent_in->debugName()].isTensor())
            {
                list_tensor.push_back((*input_map)[parent_in->debugName()].toTensor());
            }
        }
		else if (map_method_outputs.find(parent_in) != map_method_outputs.end())
        {
			list_tensor.push_back(map_method_outputs[parent_in].toTensor());
		}
        else
        {
			auto grand_node      = parent_in->node();
			auto grand_node_attr = grand_node->attributeNames()[0];

			if (strcmp(grand_node_attr.toUnqualString(), "value") == 0)
            {
				auto grand_node_kind = grand_node->kindOfS("value");

				switch (grand_node_kind)
                {
					case torch::jit::AttributeKind::i:
                        list_i.push_back(grand_node->i(grand_node_attr));
                        break;
					case torch::jit::AttributeKind::f:
                        list_f.push_back(grand_node->f(grand_node_attr));
                        break;
				}
			}
		}
	}
	if (list_i.size() == input_node->inputs().size())
    {
		torch_stack->push_back(torch::jit::IValue(list_i));
    }
	else if (list_f.size() == input_node->inputs().size())
    {
		torch_stack->push_back(torch::jit::IValue(list_f));
    }
	else
    {
		torch_stack->push_back(torch::jit::IValue(list_tensor));
    }
}

void getGetAttrParam(torch::jit::Node* input_node, torch::jit::named_attribute_list attribut_list, torch::jit::Stack* torch_stack)
{
	bool       isParameterArg = false;
	at::Tensor parameter;
	auto       parent_attr_name = input_node->s(input_node->attributeNames()[0]);
	for (const auto& param : attribut_list)
    {
		if (param.name == parent_attr_name)
        {
			isParameterArg = true;
			parameter = param.value.toTensor();
			break;
		}
	}
	torch_stack->push_back(parameter);
}

// parse string，回傳 operator type
std::string getOperatorType(const std::string& str)
{
    auto first_line_pos = str.find('\n');
    auto first_line = str.substr(0, first_line_pos);
    auto last_point_pos = str.substr(0, first_line_pos).rfind(".");
    auto operator_type = first_line.substr(last_point_pos + 1);
    return operator_type.substr(0, operator_type.length() - 2);
}

// 有 down sample 的 BasicBlock 專用
void extract_basic_block_input_output(torch::jit::NameModule module, torch::Tensor* input_tensor, int* t_a_m_r)
{
    // copy 一份 input 作為 downsample 傳入
    torch::Tensor input_copy = torch::zeros(input_tensor->sizes());
    input_copy.copy_(*input_tensor);

    for (const auto& sub_module : module.value.named_children())
    {
        if (sub_module.name == "downsample")
        {
            break;
        }
        extract_input_output(sub_module, input_tensor, t_a_m_r);
    }
    // downsample
    for (const auto& sub_module : module.value.named_children())
    {
        if (sub_module.name == "downsample")
        {
            for (const auto& downsample_module : sub_module.value.named_children())
            {
                extract_input_output(downsample_module, &input_copy, t_a_m_r);
            }
        }
    }
    // 最後一層 relu 加總
    *input_tensor = input_tensor->add(input_copy);
    // relu output activation
    *t_a_m_r += input_tensor->numel() * input_tensor->element_size();
}

void extract_input_output(torch::jit::NameModule module, torch::Tensor* input_tensor, int* t_a_m_r)
{
    // 只處理沒有 sublayer 的
    if (module.value.named_children().size() == 0)
    {
        auto operator_type = getOperatorType(module.value.dump_to_str(1, 0, 0));
        if (operator_type == "Linear" || operator_type == "Split64Linear")
        {
            auto dim = input_tensor->numel();
            *input_tensor = input_tensor->view({1, dim});
        }

        auto graph = module.value.get_method("forward").graph();
		auto nodes = graph->nodes();
        torch::jit::Stack torch_stack;
        std::map<std::string, torch::IValue> input_map;

        for (const auto& node : nodes)
        {
            // 將 tensor list 中的 tensor 分開並存入 map
            if (node->kind() == torch::prim::ListUnpack)
            {
                auto tensor_list = input_map[node->inputs()[0]->debugName()].toTensorList();
                int tensor_id = 0;
                for (const auto& tensor : tensor_list)
                {
                    input_map[node->outputs()[tensor_id++]->debugName()] = torch::IValue(tensor);
                }           
            }
            else if (node->maybeOperator())
            {
                auto operation = node->getOperation();
				auto schema    = node->schema();
                torch_stack.clear();
                
                auto input_nodes = node->inputs();
				int  idx         = 0;

                for (const auto& param : schema.arguments())
                {
					auto input_node = input_nodes[idx]->node();
                    
					switch (input_node->kind())
                    {
						case torch::prim::Constant:
							getConstantParam(input_node, param.type()->str(), &torch_stack);
							break;
						case torch::prim::ListConstruct: 
							getListParam(input_node, &torch_stack, &input_map);
							break;
						case torch::prim::GetAttr:
							getGetAttrParam(input_node, module.value.named_attributes(), &torch_stack);
							break;
						case torch::prim::Param:
							torch_stack.push_back(*input_tensor);
							break;
						default:
                            // 從 map 中取值放入 stack
                            torch_stack.push_back(input_map[input_nodes[idx]->debugName()]);
							break;
                    }

                    idx++;
				}
                // 將 output assign 為 input 移動到迴圈外面，避免像是 BatchNorm2d, Split64Linear 單 layer 中有多個 operation
                operation(torch_stack);
                // 將運算結果存回 map
                input_map[node->outputs()[0]->debugName()] = torch_stack.back();
            }
        }

        std::cout << operator_type << " " << input_tensor->sizes() << " ";
        // 將 input assign 成這層 layer 的 output
		*input_tensor = torch_stack.back().toTensor();
        std::cout << input_tensor->sizes() << std::endl;
        // 計算 output activation
        *t_a_m_r += input_tensor->numel() * input_tensor->element_size();
    }
    else
    {
        for (const auto& sub_module : module.value.named_children())
        {
            // 有 downsample 的 BasicBlock 額外處理
            auto operator_type = getOperatorType(sub_module.value.dump_to_str(1, 0, 0));
            if (operator_type == "BasicBlock" && sub_module.value.dump_to_str(1, 0, 0).find("downsample") != std::string::npos)
            {
                extract_basic_block_input_output(sub_module, input_tensor, t_a_m_r);
            }
            else
            {
                extract_input_output(sub_module, input_tensor, t_a_m_r);
            }
        }
    }
}

int main(int argc, const char* argv[])
{
    if (argc != 2)
    {
        std::cerr << "usage: ./hw2-4-2 <path-to-exported-script-module>\n";
        return -1;
    }

    torch::jit::script::Module module;
    try
    {
        module = torch::jit::load(argv[1]);
    }
    catch (const c10::Error& e)
    {
        std::cerr << "error loading the model\n";
        return -1;
    }

    torch::Tensor input_tensor = torch::randn({1, 3, 224, 224});
    int total_activation_memory_requirement = 0;
    for (const auto& sub_module : module.named_children())
    {
        extract_input_output(sub_module, &input_tensor, &total_activation_memory_requirement);
    }
    std::cout << "Total activations memory requirement: " << total_activation_memory_requirement << std::endl;

    return 0;
}
```

#### Execution Result
##### Replaced Alexnet
![](https://course.playlab.tw/md/uploads/65830dbc-aa3b-47f7-bd6c-c019ee811502.png)
##### Resnet18
![](https://course.playlab.tw/md/uploads/6c47a814-f81c-4574-ba64-32cd5d6a1c11.png)

### 2-4-3. Calculate computation requirements

#### Code
```cpp
#include <torch/script.h>

void extract_input_output(torch::jit::NameModule module, torch::Tensor* input_tensor, int* t_a_m_r);

void getConstantParam(torch::jit::Node* input_node, std::string parme_type_string, torch::jit::Stack* torch_stack)
{
	if (input_node->hasAttributes())
    {
		auto parent_attr_names = input_node->attributeNames();
		auto parent_attr       = parent_attr_names[0];

		if (strcmp(parent_attr.toUnqualString(), "value") == 0)
        {
			auto parent_attr_kind = input_node->kindOfS("value");
			switch (parent_attr_kind)
            {
				case torch::jit::AttributeKind::i:
					if (parme_type_string.substr(0, 4) == "bool")
                    {
						torch_stack->push_back(input_node->i(parent_attr) == 1);
					}
                    else
                    {
						torch_stack->push_back(input_node->i(parent_attr));
					}
					break;
				case torch::jit::AttributeKind::f:
					torch_stack->push_back(input_node->f(parent_attr));
					break;
				case torch::jit::AttributeKind::t:
					torch_stack->push_back(input_node->t(parent_attr));
					break;
                case torch::jit::AttributeKind::s:
                    if (parme_type_string.substr(0, 6) == "Device" && input_node->s(parent_attr) == "cpu")
                    {
                        torch::Device device(torch::kCPU);
                        torch_stack->push_back(torch::IValue(device));
                    }
                    break;
			}
		}
	}
    else
    {
		torch_stack->push_back(torch::jit::IValue());
	}
}

void getListParam(torch::jit::Node* input_node, torch::jit::Stack* torch_stack, std::map<std::string, torch::IValue>* input_map)
{
	std::vector<int64_t>                             list_i;
	std::vector<float>                               list_f;
	std::vector<torch::Tensor>                       list_tensor;
    std::map<torch::jit::Value*, torch::jit::IValue> map_method_outputs;

	for (const auto& parent_in : input_node->inputs())
    {
        if (input_map->find(parent_in->debugName()) != input_map->end())
        {
            if ((*input_map)[parent_in->debugName()].isInt())
            {
                list_i.push_back((*input_map)[parent_in->debugName()].toInt());
            }
            else if ((*input_map)[parent_in->debugName()].isTensor())
            {
                list_tensor.push_back((*input_map)[parent_in->debugName()].toTensor());
            }
        }
		else if (map_method_outputs.find(parent_in) != map_method_outputs.end())
        {
			list_tensor.push_back(map_method_outputs[parent_in].toTensor());
		}
        else
        {
			auto grand_node      = parent_in->node();
			auto grand_node_attr = grand_node->attributeNames()[0];

			if (strcmp(grand_node_attr.toUnqualString(), "value") == 0)
            {
				auto grand_node_kind = grand_node->kindOfS("value");

				switch (grand_node_kind)
                {
					case torch::jit::AttributeKind::i:
                        list_i.push_back(grand_node->i(grand_node_attr));
                        break;
					case torch::jit::AttributeKind::f:
                        list_f.push_back(grand_node->f(grand_node_attr));
                        break;
				}
			}
		}
	}
	if (list_i.size() == input_node->inputs().size())
    {
		torch_stack->push_back(torch::jit::IValue(list_i));
    }
	else if (list_f.size() == input_node->inputs().size())
    {
		torch_stack->push_back(torch::jit::IValue(list_f));
    }
	else
    {
		torch_stack->push_back(torch::jit::IValue(list_tensor));
    }
}

void getGetAttrParam(torch::jit::Node* input_node, torch::jit::named_attribute_list attribut_list, torch::jit::Stack* torch_stack)
{
	bool       isParameterArg = false;
	at::Tensor parameter;
	auto       parent_attr_name = input_node->s(input_node->attributeNames()[0]);
	for (const auto& param : attribut_list)
    {
		if (param.name == parent_attr_name)
        {
			isParameterArg = true;
			parameter = param.value.toTensor();
			break;
		}
	}
	torch_stack->push_back(parameter);
}

std::string getOperatorType(const std::string& str)
{
    auto first_line_pos = str.find('\n');
    auto first_line = str.substr(0, first_line_pos);
    auto last_point_pos = str.substr(0, first_line_pos).rfind(".");
    auto operator_type = first_line.substr(last_point_pos + 1);
    return operator_type.substr(0, operator_type.length() - 2);
}

void extract_basic_block_input_output(torch::jit::NameModule module, torch::Tensor* input_tensor, int* total_mac)
{
    torch::Tensor input_copy = torch::zeros(input_tensor->sizes());
    input_copy.copy_(*input_tensor);

    for (const auto& sub_module : module.value.named_children())
    {
        if (sub_module.name == "downsample")
        {
            break;
        }
        extract_input_output(sub_module, input_tensor, total_mac);
    }

    for (const auto& sub_module : module.value.named_children())
    {
        if (sub_module.name == "downsample")
        {
            for (const auto& downsample_module : sub_module.value.named_children())
            {
                extract_input_output(downsample_module, &input_copy, total_mac);
            }
        }
    }

    *input_tensor = input_tensor->add(input_copy);
    // 計算 tensor 相加的 MAC
    *total_mac += input_tensor->sizes()[0] * input_tensor->sizes()[1];
}

void extract_input_output(torch::jit::NameModule module, torch::Tensor* input_tensor, int* total_mac)
{
    if (module.value.named_children().size() == 0)
    {
        auto operator_type = getOperatorType(module.value.dump_to_str(1, 0, 0));
        if (operator_type == "Linear" || operator_type == "Split64Linear")
        {
            auto dim = input_tensor->numel();
            *input_tensor = input_tensor->view({1, dim});
        }

        auto graph = module.value.get_method("forward").graph();
		auto nodes = graph->nodes();
        torch::jit::Stack torch_stack;
        std::map<std::string, torch::IValue> input_map;

        for (const auto& node : nodes)
        {
            if (node->kind() == torch::prim::ListUnpack)
            {
                auto tensor_list = input_map[node->inputs()[0]->debugName()].toTensorList();
                int tensor_id = 0;
                for (const auto& tensor : tensor_list)
                {
                    input_map[node->outputs()[tensor_id++]->debugName()] = torch::IValue(tensor);
                }           
            }
            else if (node->maybeOperator())
            {
                auto operation = node->getOperation();
				auto schema    = node->schema();
                torch_stack.clear();
                
                auto input_nodes = node->inputs();
				int  idx         = 0;

                for (const auto& param : schema.arguments())
                {
					auto input_node = input_nodes[idx]->node();
                    
					switch (input_node->kind())
                    {
						case torch::prim::Constant:
							getConstantParam(input_node, param.type()->str(), &torch_stack);
							break;
						case torch::prim::ListConstruct: 
							getListParam(input_node, &torch_stack, &input_map);
							break;
						case torch::prim::GetAttr:
							getGetAttrParam(input_node, module.value.named_attributes(), &torch_stack);
							break;
						case torch::prim::Param:
							torch_stack.push_back(*input_tensor);
							break;
						default:
                            torch_stack.push_back(input_map[input_nodes[idx]->debugName()]);
							break;
                    }

                    idx++;
				}

                if (node->kind() == torch::aten::add)
                {
                    auto tensor1 = torch_stack[0].toTensor();
                    // 計算 tensor 相加的 MAC
                    *total_mac += tensor1.sizes()[0] * tensor1.sizes()[1];
                }
                else if (node->kind() == torch::aten::matmul)
                {
                    auto tensor1 = torch_stack[0].toTensor();
                    auto tensor2 = torch_stack[1].toTensor();
                    // 計算 tensor 相乘的 MAC
                    *total_mac += tensor1.sizes()[0] * tensor1.sizes()[1] * tensor2.sizes()[1];
                }
                
                operation(torch_stack);
                input_map[node->outputs()[0]->debugName()] = torch_stack.back();
            }
        }
        if (operator_type == "Conv2d")
        {
            // Conv2d MAC 計算
            int in_channel = input_tensor->sizes()[1];
            int kernel1, kernel2;
            for (const auto& param : module.value.named_parameters())
            {
                if (param.name.find("weight") != std::string::npos)
                {
                    kernel1 = param.value.sizes()[2];
                    kernel2 = param.value.sizes()[3];
                    break;
                }
            }
            auto kernel_ops = kernel1 * kernel2 * in_channel;

            *input_tensor = torch_stack.back().toTensor();
            
            auto dim = input_tensor->sizes();
            auto output_elements = dim[1] * dim[2] * dim[3];
            *total_mac += kernel_ops * output_elements;
        }
        else
        {
            if (operator_type == "BatchNorm2d")
            {
                // BatchNorm2d MAC 計算
                *total_mac += 2 * input_tensor->sizes()[1];
            }
            else if (operator_type == "Linear")
            {
                // Linear MAC 計算
                int in_features;
                int out_features;
                for (const auto& param : module.value.named_parameters())
                {
                    if (param.name.find("weight") != std::string::npos)
                    {
                        in_features = param.value.sizes()[1];
                        out_features = param.value.sizes()[0];
                        break;
                    }
                }

                *total_mac += in_features * out_features;
            }
            
            *input_tensor = torch_stack.back().toTensor();
        }
    }
    else
    {
        for (const auto& sub_module : module.value.named_children())
        {
            auto operator_type = getOperatorType(sub_module.value.dump_to_str(1, 0, 0));
            if (operator_type == "BasicBlock" && sub_module.value.dump_to_str(1, 0, 0).find("downsample") != std::string::npos)
            {
                extract_basic_block_input_output(sub_module, input_tensor, total_mac);
            }
            else
            {
                extract_input_output(sub_module, input_tensor, total_mac);
            }
        }
    }
}

int main(int argc, const char* argv[])
{
    if (argc != 2) {
        std::cerr << "usage: ./hw2-4-3 <path-to-exported-script-module>\n";
        return -1;
    }

    torch::jit::script::Module module;
    try {
        module = torch::jit::load(argv[1]);
    }
    catch (const c10::Error& e) {
        std::cerr << "error loading the model\n";
        return -1;
    }

    torch::Tensor input_tensor = torch::randn({1, 3, 224, 224});
    int total_mac = 0;
    for (const auto& sub_module : module.named_children())
    {
        extract_input_output(sub_module, &input_tensor, &total_mac);
    }
    std::cout << "Total MAC: " << total_mac << std::endl;

    return 0;
}
```

#### Execution Result
##### Replaced Alexnet
![](https://course.playlab.tw/md/uploads/f1b4e614-cd68-4c56-87e9-4994c43a309f.png)
##### Resnet18
![](https://course.playlab.tw/md/uploads/77a295e3-65b5-4806-9583-96296664f772.png)

### 2-4-4. Compare your results to the result in HW2-1 and HW2-2

#### Discussion
1. 在2-4-1的部分計算結果(244,403,360)與torchinfo得到的結果(244 MB, Total params: 61,100,840)相同 (61,100,840 * 4 = 244,403,360)
2. 在2-4-2的部分計算結果與2-2-3計算結果相比有落差，但是由於2-2-3是以會釋放不需要的activation儲存空間為前提計算，2-4-2中是計算所有activation儲存空間，因此兩者結果不同是合理的
3. 在2-4-3的部分計算結果(715,113,640)與torchinfo得到的結果(714.68 M)相近(約為681.99M)，推測因為部分像是ReLU、MaxPool2d layer的MAC並未計算，導致結果可能有小幅度落差

##### Replaced Alexnet
```
===================================================================================================================
Layer (type:depth-idx)                   Input Shape               Output Shape              Param #
===================================================================================================================
AlexNet                                  [1, 3, 224, 224]          [1, 1000]                 --
├─Sequential: 1-1                        [1, 3, 224, 224]          [1, 256, 6, 6]            --
│    └─Conv2d: 2-1                       [1, 3, 224, 224]          [1, 64, 55, 55]           23,296
│    └─ReLU: 2-2                         [1, 64, 55, 55]           [1, 64, 55, 55]           --
│    └─MaxPool2d: 2-3                    [1, 64, 55, 55]           [1, 64, 27, 27]           --
│    └─Conv2d: 2-4                       [1, 64, 27, 27]           [1, 192, 27, 27]          307,392
│    └─ReLU: 2-5                         [1, 192, 27, 27]          [1, 192, 27, 27]          --
│    └─MaxPool2d: 2-6                    [1, 192, 27, 27]          [1, 192, 13, 13]          --
│    └─Conv2d: 2-7                       [1, 192, 13, 13]          [1, 384, 13, 13]          663,936
│    └─ReLU: 2-8                         [1, 384, 13, 13]          [1, 384, 13, 13]          --
│    └─Conv2d: 2-9                       [1, 384, 13, 13]          [1, 256, 13, 13]          884,992
│    └─ReLU: 2-10                        [1, 256, 13, 13]          [1, 256, 13, 13]          --
│    └─Conv2d: 2-11                      [1, 256, 13, 13]          [1, 256, 13, 13]          590,080
│    └─ReLU: 2-12                        [1, 256, 13, 13]          [1, 256, 13, 13]          --
│    └─MaxPool2d: 2-13                   [1, 256, 13, 13]          [1, 256, 6, 6]            --
├─AdaptiveAvgPool2d: 1-2                 [1, 256, 6, 6]            [1, 256, 6, 6]            --
├─Sequential: 1-3                        [1, 9216]                 [1, 1000]                 --
│    └─Dropout: 2-14                     [1, 9216]                 [1, 9216]                 --
│    └─Split64Linear: 2-15               [1, 9216]                 [1, 4096]                 37,752,832
│    └─ReLU: 2-16                        [1, 4096]                 [1, 4096]                 --
│    └─Dropout: 2-17                     [1, 4096]                 [1, 4096]                 --
│    └─Split64Linear: 2-18               [1, 4096]                 [1, 4096]                 16,781,312
│    └─ReLU: 2-19                        [1, 4096]                 [1, 4096]                 --
│    └─Split64Linear: 2-20               [1, 4096]                 [1, 1000]                 4,097,000
===================================================================================================================
Total params: 61,100,840
Trainable params: 61,100,840
Non-trainable params: 0
Total mult-adds (M): 714.68
===================================================================================================================
Input size (MB): 0.60
Forward/backward pass size (MB): 3.95
Params size (MB): 244.40
Estimated Total Size (MB): 248.96
===================================================================================================================
```
##### Resnet18
```
===================================================================================================================
Layer (type:depth-idx)                   Input Shape               Output Shape              Param #
===================================================================================================================
ResNet                                   [1, 3, 224, 224]          [1, 1000]                 --
├─Conv2d: 1-1                            [1, 3, 224, 224]          [1, 64, 112, 112]         9,408
├─BatchNorm2d: 1-2                       [1, 64, 112, 112]         [1, 64, 112, 112]         128
├─ReLU: 1-3                              [1, 64, 112, 112]         [1, 64, 112, 112]         --
├─MaxPool2d: 1-4                         [1, 64, 112, 112]         [1, 64, 56, 56]           --
├─Sequential: 1-5                        [1, 64, 56, 56]           [1, 64, 56, 56]           --
│    └─BasicBlock: 2-1                   [1, 64, 56, 56]           [1, 64, 56, 56]           --
│    │    └─Conv2d: 3-1                  [1, 64, 56, 56]           [1, 64, 56, 56]           36,864
│    │    └─BatchNorm2d: 3-2             [1, 64, 56, 56]           [1, 64, 56, 56]           128
│    │    └─ReLU: 3-3                    [1, 64, 56, 56]           [1, 64, 56, 56]           --
│    │    └─Conv2d: 3-4                  [1, 64, 56, 56]           [1, 64, 56, 56]           36,864
│    │    └─BatchNorm2d: 3-5             [1, 64, 56, 56]           [1, 64, 56, 56]           128
│    │    └─ReLU: 3-6                    [1, 64, 56, 56]           [1, 64, 56, 56]           --
│    └─BasicBlock: 2-2                   [1, 64, 56, 56]           [1, 64, 56, 56]           --
│    │    └─Conv2d: 3-7                  [1, 64, 56, 56]           [1, 64, 56, 56]           36,864
│    │    └─BatchNorm2d: 3-8             [1, 64, 56, 56]           [1, 64, 56, 56]           128
│    │    └─ReLU: 3-9                    [1, 64, 56, 56]           [1, 64, 56, 56]           --
│    │    └─Conv2d: 3-10                 [1, 64, 56, 56]           [1, 64, 56, 56]           36,864
│    │    └─BatchNorm2d: 3-11            [1, 64, 56, 56]           [1, 64, 56, 56]           128
│    │    └─ReLU: 3-12                   [1, 64, 56, 56]           [1, 64, 56, 56]           --
├─Sequential: 1-6                        [1, 64, 56, 56]           [1, 128, 28, 28]          --
│    └─BasicBlock: 2-3                   [1, 64, 56, 56]           [1, 128, 28, 28]          --
│    │    └─Conv2d: 3-13                 [1, 64, 56, 56]           [1, 128, 28, 28]          73,728
│    │    └─BatchNorm2d: 3-14            [1, 128, 28, 28]          [1, 128, 28, 28]          256
│    │    └─ReLU: 3-15                   [1, 128, 28, 28]          [1, 128, 28, 28]          --
│    │    └─Conv2d: 3-16                 [1, 128, 28, 28]          [1, 128, 28, 28]          147,456
│    │    └─BatchNorm2d: 3-17            [1, 128, 28, 28]          [1, 128, 28, 28]          256
│    │    └─Sequential: 3-18             [1, 64, 56, 56]           [1, 128, 28, 28]          --
│    │    │    └─Conv2d: 4-1             [1, 64, 56, 56]           [1, 128, 28, 28]          8,192
│    │    │    └─BatchNorm2d: 4-2        [1, 128, 28, 28]          [1, 128, 28, 28]          256
│    │    └─ReLU: 3-19                   [1, 128, 28, 28]          [1, 128, 28, 28]          --
│    └─BasicBlock: 2-4                   [1, 128, 28, 28]          [1, 128, 28, 28]          --
│    │    └─Conv2d: 3-20                 [1, 128, 28, 28]          [1, 128, 28, 28]          147,456
│    │    └─BatchNorm2d: 3-21            [1, 128, 28, 28]          [1, 128, 28, 28]          256
│    │    └─ReLU: 3-22                   [1, 128, 28, 28]          [1, 128, 28, 28]          --
│    │    └─Conv2d: 3-23                 [1, 128, 28, 28]          [1, 128, 28, 28]          147,456
│    │    └─BatchNorm2d: 3-24            [1, 128, 28, 28]          [1, 128, 28, 28]          256
│    │    └─ReLU: 3-25                   [1, 128, 28, 28]          [1, 128, 28, 28]          --
├─Sequential: 1-7                        [1, 128, 28, 28]          [1, 256, 14, 14]          --
│    └─BasicBlock: 2-5                   [1, 128, 28, 28]          [1, 256, 14, 14]          --
│    │    └─Conv2d: 3-26                 [1, 128, 28, 28]          [1, 256, 14, 14]          294,912
│    │    └─BatchNorm2d: 3-27            [1, 256, 14, 14]          [1, 256, 14, 14]          512
│    │    └─ReLU: 3-28                   [1, 256, 14, 14]          [1, 256, 14, 14]          --
│    │    └─Conv2d: 3-29                 [1, 256, 14, 14]          [1, 256, 14, 14]          589,824
│    │    └─BatchNorm2d: 3-30            [1, 256, 14, 14]          [1, 256, 14, 14]          512
│    │    └─Sequential: 3-31             [1, 128, 28, 28]          [1, 256, 14, 14]          --
│    │    │    └─Conv2d: 4-3             [1, 128, 28, 28]          [1, 256, 14, 14]          32,768
│    │    │    └─BatchNorm2d: 4-4        [1, 256, 14, 14]          [1, 256, 14, 14]          512
│    │    └─ReLU: 3-32                   [1, 256, 14, 14]          [1, 256, 14, 14]          --
│    └─BasicBlock: 2-6                   [1, 256, 14, 14]          [1, 256, 14, 14]          --
│    │    └─Conv2d: 3-33                 [1, 256, 14, 14]          [1, 256, 14, 14]          589,824
│    │    └─BatchNorm2d: 3-34            [1, 256, 14, 14]          [1, 256, 14, 14]          512
│    │    └─ReLU: 3-35                   [1, 256, 14, 14]          [1, 256, 14, 14]          --
│    │    └─Conv2d: 3-36                 [1, 256, 14, 14]          [1, 256, 14, 14]          589,824
│    │    └─BatchNorm2d: 3-37            [1, 256, 14, 14]          [1, 256, 14, 14]          512
│    │    └─ReLU: 3-38                   [1, 256, 14, 14]          [1, 256, 14, 14]          --
├─Sequential: 1-8                        [1, 256, 14, 14]          [1, 512, 7, 7]            --
│    └─BasicBlock: 2-7                   [1, 256, 14, 14]          [1, 512, 7, 7]            --
│    │    └─Conv2d: 3-39                 [1, 256, 14, 14]          [1, 512, 7, 7]            1,179,648
│    │    └─BatchNorm2d: 3-40            [1, 512, 7, 7]            [1, 512, 7, 7]            1,024
│    │    └─ReLU: 3-41                   [1, 512, 7, 7]            [1, 512, 7, 7]            --
│    │    └─Conv2d: 3-42                 [1, 512, 7, 7]            [1, 512, 7, 7]            2,359,296
│    │    └─BatchNorm2d: 3-43            [1, 512, 7, 7]            [1, 512, 7, 7]            1,024
│    │    └─Sequential: 3-44             [1, 256, 14, 14]          [1, 512, 7, 7]            --
│    │    │    └─Conv2d: 4-5             [1, 256, 14, 14]          [1, 512, 7, 7]            131,072
│    │    │    └─BatchNorm2d: 4-6        [1, 512, 7, 7]            [1, 512, 7, 7]            1,024
│    │    └─ReLU: 3-45                   [1, 512, 7, 7]            [1, 512, 7, 7]            --
│    └─BasicBlock: 2-8                   [1, 512, 7, 7]            [1, 512, 7, 7]            --
│    │    └─Conv2d: 3-46                 [1, 512, 7, 7]            [1, 512, 7, 7]            2,359,296
│    │    └─BatchNorm2d: 3-47            [1, 512, 7, 7]            [1, 512, 7, 7]            1,024
│    │    └─ReLU: 3-48                   [1, 512, 7, 7]            [1, 512, 7, 7]            --
│    │    └─Conv2d: 3-49                 [1, 512, 7, 7]            [1, 512, 7, 7]            2,359,296
│    │    └─BatchNorm2d: 3-50            [1, 512, 7, 7]            [1, 512, 7, 7]            1,024
│    │    └─ReLU: 3-51                   [1, 512, 7, 7]            [1, 512, 7, 7]            --
├─AdaptiveAvgPool2d: 1-9                 [1, 512, 7, 7]            [1, 512, 1, 1]            --
├─Linear: 1-10                           [1, 512]                  [1, 1000]                 513,000
===================================================================================================================
Total params: 11,689,512
Trainable params: 11,689,512
Non-trainable params: 0
Total mult-adds (G): 1.81
===================================================================================================================
Input size (MB): 0.60
Forward/backward pass size (MB): 39.75
Params size (MB): 46.76
Estimated Total Size (MB): 87.11
===================================================================================================================
```
