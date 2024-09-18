import torch
import torch.nn as nn

class Subgraph2(nn.Module):
    def __init__(self):
        super(Subgraph2, self).__init__()
    
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
                if len(split_row_a) != len(split_col_b):
                    raise ValueError
                for i in range(len(split_row_a)):
                    mul_tensor = torch.add(mul_tensor, torch.matmul(split_row_a[i], split_col_b[i]))
                rowwise_tensors.append(mul_tensor)
            total_tensors.append(rowwise_tensors)
        return total_tensors

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