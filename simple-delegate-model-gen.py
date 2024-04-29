import torch
import numpy as np
from torch.autograd import Variable
from torch import nn


def create_test_model_for_delegate_one_input():
    def form_linear(weights, bias):
        weights = np.array(weights)
        bias = np.array(bias)
        layer = nn.Linear(weights.shape[1], weights.shape[0])
        layer.weight = nn.Parameter(
            torch.tensor(
                weights,
                dtype=torch.float32,
                requires_grad=False
            ),
            requires_grad=False
        )
        layer.bias = nn.Parameter(
            torch.tensor(
                bias,
                dtype=torch.float32,
                requires_grad=False
            ),
            requires_grad=False
        )
        return layer

    class TestDelegateModel(torch.nn.Module):
        def __init__(self):
            super(TestDelegateModel, self).__init__()
            self.l = {}  # noqa: E741
            self.l['x1'] = form_linear(
                weights=[
                    [1, 0, 2, 4],
                    [2, -1, 2, -7],
                    [1, 3, 1, -7]
                ],
                bias=[1, 4, 3]
            )
            self.l['x2'] = form_linear(
                weights=[
                    [1, 7, 2],
                    [8, 4, -5],
                    [6, 2, -1]
                ],
                bias=[-5, 8, 7]
            )
            self.l['x3'] = form_linear(
                weights=[
                    [3, 7, 2],
                    [-1, -5, 8],
                ],
                bias=[2, 4]
            )
            self.l['r1'] = form_linear(
                weights=[
                    [-1, 3],
                ],
                bias=[6]
            )

            self.relu = nn.ReLU()

        def forward(self, x):
            rx1 = self.relu(self.l['x1'](x))
            rx2 = self.relu(self.l['x2'](rx1))
            rx3 = rx1 + rx2
            rx4 = self.relu(self.l['x3'](rx3))
            return self.relu(self.l['r1'](rx4))

    model = TestDelegateModel()
    data = Variable(torch.FloatTensor([[1, 0, 0.5, -1.5]]))
    print(model.forward(data))
    torch.onnx.export(
        model,
        (data,),
        "test-delegate-one-input.onnx",
        input_names=["input"],
        output_names=["output"]
    )


def create_test_model_for_delegate_two_inputs():
    def form_linear(weights, bias):
        weights = np.array(weights)
        bias = np.array(bias)
        layer = nn.Linear(weights.shape[1], weights.shape[0])
        layer.weight = nn.Parameter(
            torch.tensor(
                weights,
                dtype=torch.float32,
                requires_grad=False
            ),
            requires_grad=False
        )
        layer.bias = nn.Parameter(
            torch.tensor(
                bias,
                dtype=torch.float32,
                requires_grad=False
            ),
            requires_grad=False
        )
        return layer

    class TestDelegateModel(torch.nn.Module):
        def __init__(self):
            super(TestDelegateModel, self).__init__()
            self.l = {}  # noqa: E741
            self.l['x1'] = form_linear(
                weights=[
                    [1, 0, 2, 4],
                    [2, -1, 2, -7],
                    [1, 3, 1, -7]
                ],
                bias=[1, 4, 3]
            )
            self.l['x2'] = form_linear(
                weights=[
                    [3, 7, 2],
                    [-1, -5, 8],
                ],
                bias=[2, 4]
            )
            self.l['y1'] = form_linear(
                weights=[
                    [1, 7, 2],
                    [8, 4, -5]
                ],
                bias=[-5, 8]
            )
            self.l['r1'] = form_linear(
                weights=[
                    [-1, 3],
                ],
                bias=[6]
            )

            self.relu = nn.ReLU()

        def forward(self, x, y):
            rx1 = self.relu(self.l['x1'](x))
            rx2 = self.relu(self.l['x2'](rx1))
            ry1 = self.relu(self.l['y1'](y))
            rs1 = rx2 + ry1
            return self.relu(self.l['r1'](rs1))

    model = TestDelegateModel()
    data1 = Variable(torch.FloatTensor([[1, 0, 0.5, -1.5]]))
    data2 = Variable(torch.FloatTensor([[4.5, -0.5, 9]]))
    print(model.forward(data1, data2))
    torch.onnx.export(
        model,
        (data1, data2,),
        "test-delegate-two-inputs.onnx",
        input_names=["input_1", "input_2"],
        output_names=["output"]
    )


if __name__ == '__main__':
    create_test_model_for_delegate_one_input()
    create_test_model_for_delegate_two_inputs()
