import torch
from torch import nn
import torch.nn.functional as F


class SRCNN(nn.Module):
    def __init__(self, kernel_pattern=None, channel_pattern=None, padding='same'):
        """
        Basic SRCNN network.
        :param kernel_pattern: list of kernel sizes selected for the network.
        :param channel_pattern: list of channel i/o count for the network.
        :param padding: Specify type of image padding for the network.
        """
        super(SRCNN, self).__init__()

        if kernel_pattern is None:
            kernel_pattern = [9, 5, 5]  # default SRCNN
        if channel_pattern is None:
            channel_pattern = [1, 64, 32, 1]  # default SRCNN

        channel_sequence = []
        for index, c in enumerate(channel_pattern[:-1]):
            channel_sequence.append((c, channel_pattern[index+1]))

        if padding == 'same':  # input and output will be the same size through the network
            padding = [k//2 for k in kernel_pattern]
        else:
            padding = 0

        self.layer_dict = nn.ModuleDict()  # this stores all layers in the network
        self.depth = len(kernel_pattern)

        self.build_network(kernel_pattern, channel_sequence, padding)

    def build_network(self, kernel_pattern, channel_sequence, padding):
        """
        This function builds up the network based on given parameters.
        :param kernel_pattern: list of CNN kernel sizes in input->output order.
        :param channel_sequence: list of CNN input/output channels in input->output order.
        :param padding: list of padding values for CNN in input->output order.
        :return:
        """
        for index, (k, p, c) in enumerate(zip(kernel_pattern, padding, channel_sequence)):
            # conv output = n + 2p - f + 1 x n + 2p -f + 1
            self.layer_dict['conv_{}'.format(index)] = nn.Conv2d(c[0], c[1], kernel_size=k, padding=p)  # bias implicit

    def forward(self, x):
        for i in range(self.depth):
            x = self.layer_dict['conv_{}'.format(i)].forward(x)
            if i != self.depth-1:
                x = F.relu(x)
        return x

    def reset_parameters(self):
        """
        This functions resets all the weights of the network.
        :return: None
        """
        for layer in self.layer_dict.children():
            layer.reset_parameters()  # resets weights to uniform distribution


class VDSR(SRCNN):
    """
    Deeper version of SRCNN.  Also features a residual forward network function.
    """
    def forward(self, x):
        residual = x
        for i in range(self.depth):
            if i == 0:
                out = self.layer_dict['conv_{}'.format(i)].forward(x)
            else:
                out = self.layer_dict['conv_{}'.format(i)].forward(out)
            if i != self.depth-1:
                out = F.relu(out)

        return torch.add(out, residual)
