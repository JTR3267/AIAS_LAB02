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