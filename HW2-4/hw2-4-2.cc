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