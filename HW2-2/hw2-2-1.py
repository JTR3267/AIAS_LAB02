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