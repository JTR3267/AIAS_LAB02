#include <torch/script.h>

void getConstantParam(torch::jit::Node* input_node, std::string parme_type_string, torch::jit::Stack* torch_stack) {
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

void getListParam(torch::jit::Node* input_node, torch::jit::Stack* torch_stack) {
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

void getGetAttrParam(torch::jit::Node* input_node, torch::jit::named_attribute_list attribut_list, torch::jit::Stack* torch_stack) {
	bool       isParameterArg = false;
	at::Tensor parameter;
	auto       parent_attr_name = input_node->s(input_node->attributeNames()[0]);
	for (const auto& param : attribut_list) {
		if (param.name == parent_attr_name) {
			isParameterArg = true;
			parameter = param.value.toTensor();
			break;
		}
	}
	torch_stack->push_back(parameter);
}

void extract_input_output(torch::jit::NameModule module, torch::Tensor* input_tensor, int* t_a_m_r)
{
    if (module.value.named_children().size() == 0)
    {
        if (module.value.dump_to_str(1, 0, 0).find("BatchNorm2d") != std::string::npos)
        {
            module.name == "1" ? (std::cout << "bn ") : (std::cout << module.name << " ");
            std::cout << input_tensor->sizes() << " " << input_tensor->sizes() << std::endl;
            return;
        }
        else if (module.value.dump_to_str(1, 0, 0).find("Linear") != std::string::npos)
        {
            auto dim = input_tensor->numel();
            *input_tensor = input_tensor->view({1, dim});
        }
        else if (module.name == "0")
        {
            auto input_shape = input_tensor->sizes();
            auto dtype = input_tensor->dtype();
            *input_tensor = torch::randn({input_shape[0], input_shape[1] / 2, input_shape[2] * 2, input_shape[3] * 2}).to(dtype);
        }
        
        module.name == "0" ? (std::cout << "conv ") : (std::cout << module.name << " ");
        std::cout << input_tensor->sizes() << " ";

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
							std::cout << "" << input_node->kind() << std::endl;
							break;
					}					
				}
                operation(torch_stack);   
				*input_tensor = torch_stack.back().toTensor();
            }
        }
        std::cout << input_tensor->sizes() << std::endl;
        *t_a_m_r += input_tensor->numel() * input_tensor->element_size();
    }
    else
    {
        for (const auto& sub_module : module.value.named_children())
        {
            extract_input_output(sub_module, input_tensor, t_a_m_r);
        }
    }
}

int main(int argc, const char* argv[]) {
    if (argc != 2) {
        std::cerr << "usage: ./hw2-4-2 <path-to-exported-script-module>\n";
        return -1;
    }

    torch::jit::script::Module module;
    try {
        module = torch::jit::load(argv[1]);
        torch::Tensor input_tensor = torch::randn({1, 3, 224, 224});
        int total_activation_memory_requirement = 0;
        for (const auto& sub_module : module.named_children())
        {
            extract_input_output(sub_module, &input_tensor, &total_activation_memory_requirement);
        }
        std::cout << "Total activations memory requirement: " << total_activation_memory_requirement << std::endl;
    }
    catch (const c10::Error& e) {
        std::cerr << "error loading the model\n";
        return -1;
    }

    return 0;
}
