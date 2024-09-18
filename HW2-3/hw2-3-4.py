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
    
def compare_tensor(tensor1, tensor2):
    if tensor1.shape != tensor2.shape:
        print(f"Output shape different")
    else:
        print(f"Same output shape {tensor1.shape}")
        
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
        linear_tensor = super(CompareSplit64Linear, self).forward(input_tensor)
        print("Linear layer output compare")
        compare_tensor(concat_tensor, linear_tensor)
        return concat_tensor

alexnet_input = torch.randn(10, 3, 224, 224)
alexnet = models.alexnet(pretrained=True)
replaced_alexnet = copy.deepcopy(alexnet)
replace_linear_layer(replaced_alexnet, CompareSplit64Linear)
alexnet.eval()
replaced_alexnet.eval()
original_output = alexnet(alexnet_input)
replaced_output = replaced_alexnet(alexnet_input)
print("Model output compare")
compare_tensor(original_output, replaced_output)