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

input_bandwith = 0
output_bandwith = 0

for i in onnx_model.graph.node:
    for input_name in i.input:
        input_bandwith += mutiply_tuple(get_value_by_key(input_name))
    for output_name in i.output:
        output_bandwith += mutiply_tuple(get_value_by_key(output_name))

total_bandwith = input_bandwith + output_bandwith

print(f"Data bandwidth requirement: {total_bandwith}")