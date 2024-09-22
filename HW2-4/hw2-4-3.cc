#include <torch/script.h>
#include <torch/torch.h>

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
			}
		}
	}
    else
    {
		torch_stack->push_back(torch::jit::IValue());
	}
}

void getListParam(torch::jit::Node* input_node, torch::jit::Stack* torch_stack)
{
	std::vector<int64_t>                             list_i;
	std::vector<float>                               list_f;
	std::vector<torch::Tensor>                       list_tensor;
    std::map<torch::jit::Value*, torch::jit::IValue> map_method_outputs;

	for (const auto& parent_in : input_node->inputs())
    {
		if (map_method_outputs.find(parent_in) != map_method_outputs.end())
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

void extract_input_output(torch::jit::NameModule module, torch::Tensor* input_tensor, int* total_mac)
{
    if (module.value.named_children().size() == 0)
    {
        auto operator_type = getOperatorType(module.value.dump_to_str(1, 0, 0));

        if (operator_type == "BatchNorm2d")
        {
            *total_mac += 2 * input_tensor->sizes()[1];
            return;
        }
        else if (operator_type == "Split64Linear")
        {
            auto dim = input_tensor->numel();
            *input_tensor = input_tensor->view({1, dim});

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
            auto linear_layer = torch::nn::Linear(torch::nn::LinearOptions(in_features, out_features));
            *input_tensor = linear_layer(*input_tensor);

            *total_mac += in_features * out_features;
            return;
        }
        else if (operator_type == "Linear")
        {
            auto dim = input_tensor->numel();
            *input_tensor = input_tensor->view({1, dim});

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

        auto graph = module.value.get_method("forward").graph();
		auto nodes = graph->nodes();

        for (const auto& node : nodes)
        {
            if (node->maybeOperator())
            {
                auto operation = node->getOperation();
				auto schema    = node->schema();
                torch::jit::Stack torch_stack;
                
                auto input_nodes = node->inputs();
				int  idx         = 0;

                for (const auto& param : schema.arguments())
                {
					auto input_node = input_nodes[idx++]->node();
                    
					switch (input_node->kind())
                    {
						case torch::prim::Constant:
							getConstantParam(input_node, param.type()->str(), &torch_stack);
							break;
						case torch::prim::ListConstruct: 
							getListParam(input_node, &torch_stack);
							break;
						case torch::prim::GetAttr:
							getGetAttrParam(input_node, module.value.named_attributes(), &torch_stack);
							break;
						case torch::prim::Param:
							torch_stack.push_back(*input_tensor);
							break;
						default:
							std::cout << "Other kind of input node: " << input_node->kind() << std::endl;
							break;
					}					
				}
                try
                {
                    if (operator_type == "Conv2d")
                    {
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

                        operation(torch_stack);
				        *input_tensor = torch_stack.back().toTensor();
                        
                        auto dim = input_tensor->sizes();
                        auto output_elements = dim[1] * dim[2] * dim[3];
                        *total_mac += kernel_ops * output_elements;
                    }
                    else
                    {
                        operation(torch_stack);
				        *input_tensor = torch_stack.back().toTensor();
                    }
                }
                catch(const std::exception& e)
                {
                    if (operator_type == "Conv2d")
                    {
                        auto dim = input_tensor->sizes();
                        *input_tensor = torch::randn({dim[0], dim[1] / 2, dim[2] * 2, dim[3] * 2});
                        return extract_input_output(module, input_tensor, total_mac);
                    }
                }
            }
        }
    }
    else
    {
        for (const auto& sub_module : module.value.named_children())
        {
            extract_input_output(sub_module, input_tensor, total_mac);
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
        torch::Tensor input_tensor = torch::randn({1, 3, 224, 224});
        int total_mac = 0;
        for (const auto& sub_module : module.named_children())
        {
            extract_input_output(sub_module, &input_tensor, &total_mac);
        }
        std::cout << "Total MAC: " << total_mac << std::endl;
    }
    catch (const c10::Error& e) {
        std::cerr << "error loading the model\n";
        return -1;
    }

    return 0;
}