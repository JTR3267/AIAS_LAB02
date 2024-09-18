import torch
import torch.nn as nn
import torchvision.models as models

def replace_linear_layer(model, new_module):
    for child_name, child_module in model.named_children():
        if isinstance(child_module, nn.Linear):
            setattr(model, child_name, new_module(child_module.in_features, child_module.out_features))
        else:
            replace_linear_layer(child_module, new_module)

class Split64Linear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True):
        super(Split64Linear, self).__init__(in_features, out_features, bias)
    
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
        return concat_tensor

alexnet = models.alexnet(pretrained=True)
replace_linear_layer(alexnet, Split64Linear)
alexnet_input = torch.randn(1, 3, 224, 224)
torch.onnx.export(alexnet, alexnet_input, 'replaced_alexnet.onnx', input_names=['input'], output_names=['output'])