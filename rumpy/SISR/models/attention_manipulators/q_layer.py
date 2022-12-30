from torch import nn
import torch


class ParaCALayer(nn.Module):
    """
    Main Meta-Attention module.

    This module accepts as input both the current set of channels within a CNN and a vector containing
    relevant metadata for the image under analysis.  The metadata vector will be used to modulate the network channels,
     using a channel attention scheme.
    """
    def __init__(self, network_channels, num_metadata, nonlinearity=False, num_layers=2, dropout=False, dropout_probability=None):
        """
        :param network_channels: Number of feature channels expected in network.
        :param num_metadata: Metadata vector size.
        :param nonlinearity: Set to True to add ReLUs between each FC layer in the module.
        :param num_layers: Number of fully-connected layers to introduce.  With 2 or more layers, the unit size
        is increased/decreased consistently from the input to the final modulation vector.
        """
        super(ParaCALayer, self).__init__()

        layers = []
        multiplier = num_layers
        inputs = [num_metadata]

        for i in range(num_layers):
            if num_metadata > 15:
                inputs.append((network_channels-num_metadata)//multiplier + num_metadata)
            else:
                inputs.append(network_channels//multiplier)
            layers.append(nn.Conv2d(inputs[i], inputs[i+1], 1, padding=0, bias=True))
            if nonlinearity and multiplier != 1:
                layers.append(nn.ReLU(inplace=True))
            if dropout and multiplier != 1:
                layers.append(nn.Dropout(p=dropout_probability))
            multiplier -= 1

        layers.append(nn.Sigmoid())
        self.attribute_integrator = nn.Sequential(*layers)

    def forward(self, x, attributes):

        y = self.attribute_integrator(attributes)

        return x * y

    def forensic(self, x, attributes):

        y = self.attribute_integrator(attributes)

        return x * y, x.cpu().data.numpy().squeeze(), (x * y).cpu().data.numpy().squeeze(), y.cpu().data.numpy().squeeze()


class ResPipesCALayer(nn.Module):
    """
    Residual Meta-Attention module
    """
    def __init__(self, network_channels, num_metadata, nonlinearity=False, num_layers=2, num_pipes=3, combine_pipes='concat'):
        super(ResPipesCALayer, self).__init__()

        start = num_metadata
        stop = network_channels

        self.num_pipes = num_pipes
        self.combine_pipes = combine_pipes
        self.pipes = nn.ModuleList()
        for i in range(num_pipes):
            pipe = []

            num_pipe_layers = 0

            if isinstance(num_layers, list):
                num_pipe_layers = num_layers[i]
            else:
                num_pipe_layers = num_layers + i

            num_layer_sizes = num_pipe_layers + 1

            # Create a linspace from start to stop using num_layer_sizes as the number of points
            diff = (stop - start)/(num_layer_sizes - 1)
            layer_sizes = [int(diff * i + start)  for i in range(num_layer_sizes)]

            for j in range(num_pipe_layers):
                pipe.append(nn.Conv2d(layer_sizes[j], layer_sizes[j+1], 1, padding=0, bias=True))
                if nonlinearity:
                    pipe.append(nn.ReLU(inplace=True))

            seq_pipe = nn.Sequential(*pipe)
            self.pipes.append(seq_pipe)

        final_layers = []

        if combine_pipes == 'add':
            final_layers.append(nn.Conv2d(network_channels, network_channels, 1, padding=0, bias=True))
        else:
            final_layers.append(nn.Conv2d(network_channels*num_pipes, network_channels, 1, padding=0, bias=True))
        final_layers.append(nn.Sigmoid())

        self.attention_vector = nn.Sequential(*final_layers)

    def forward(self, x, attributes):
        combined_pipes = torch.cat([self.pipes[i](attributes) for i in range(self.num_pipes)], dim=1)

        if self.combine_pipes == 'add':
            stacked_pipes = torch.stack([self.pipes[i](attributes) for i in range(self.num_pipes)])
            combined_pipes = torch.sum(stacked_pipes, dim=0)

        y = self.attention_vector(combined_pipes)

        return x * y

    def forensic(self, x, attributes):
        combined_pipes = torch.cat([self.pipes[i](attributes) for i in range(self.num_pipes)], dim=1)

        if self.combine_pipes == 'add':
            stacked_pipes = torch.stack([self.pipes[i](attributes) for i in range(self.num_pipes)])
            combined_pipes = torch.sum(stacked_pipes, dim=0)

        y = self.attention_vector(combined_pipes)

        return x * y, y.cpu().data.numpy().squeeze()


class ResPipesSplitCALayer(nn.Module):
    """
    Residual Meta-Attention module
    """
    def __init__(self, network_channels, num_metadata, nonlinearity=False, num_layers=2, num_pipes=3, split_percent=0.25):
        super(ResPipesSplitCALayer, self).__init__()

        split_features = int(network_channels * split_percent)
        remainder_features = network_channels - split_features

        self.num_pipes = num_pipes
        self.split_percent = split_percent
        self.split_features = split_features
        self.remainder_features = remainder_features

        self.pipes = nn.ModuleList()
        for i in range(num_pipes):
            pipe = []

            num_pipe_layers = 0

            if isinstance(num_layers, list):
                num_pipe_layers = num_layers[i]
            else:
                num_pipe_layers = num_layers + i

            num_layer_sizes = num_pipe_layers + 1

            start = num_metadata
            stop = network_channels

            if i > 0:
                start = remainder_features

            if i == (num_pipes - 1):
                stop = split_features

            # Create a linspace from start to stop using num_layer_sizes as the number of points
            diff = (stop - start)/(num_layer_sizes - 1)
            layer_sizes = [int(diff * i + start)  for i in range(num_layer_sizes)]

            for j in range(num_pipe_layers):
                pipe.append(nn.Conv2d(layer_sizes[j], layer_sizes[j+1], 1, padding=0, bias=True))

                if nonlinearity:
                    pipe.append(nn.ReLU(inplace=True))

            seq_pipe = nn.Sequential(*pipe)
            self.pipes.append(seq_pipe)

        final_layers = []

        final_layers.append(nn.Conv2d(split_features*num_pipes, network_channels, 1, padding=0, bias=True))
        final_layers.append(nn.Sigmoid())

        self.attention_vector = nn.Sequential(*final_layers)

    def forward(self, x, attributes):
        # attributes -> self.pipes[0] -> tensor_p0 of size network_channels
        # tensor_p0 -> split into -> tensor_p0_a of size (int(network_channels * user chosen value)) and tensor_p0_b (remaining)
        # tensor_p0_b -> self.pipes[1] -> tensor_p1 of size network channels

        # possible code:
        # processed = int(network_channels * user chosen value)
        # remaining = network_channels - processed
        # processed_remaining = [torch.split(self.pipe[i], [processed, remaining]) for i in range(self.num_pipes)]

        pipe_0_out_full = self.pipes[0](attributes)
        pipe_0_out_split, pipe_0_out_remainder = torch.split(pipe_0_out_full, [self.split_features, self.remainder_features], dim=1)

        pipe_outputs_full = [pipe_0_out_full]
        pipe_outputs_split = [pipe_0_out_split]
        pipe_outputs_remainder = [pipe_0_out_remainder]

        for i in range(1, self.num_pipes):
            pipe_outputs_full.append(self.pipes[i](pipe_outputs_remainder[i-1]))
            if i == (self.num_pipes - 1):
                pipe_outputs_split.append(pipe_outputs_full[i])
            else:
                pipe_outputs_split.append(torch.split(pipe_outputs_full[i], [self.split_features, self.remainder_features], dim=1)[0])
                pipe_outputs_remainder.append(torch.split(pipe_outputs_full[i], [self.split_features, self.remainder_features], dim=1)[1])

        combined_pipes = torch.cat(pipe_outputs_split, dim=1)

        y = self.attention_vector(combined_pipes)

        return x * y

    def forensic(self, x, attributes):
        pipe_0_out_full = self.pipes[0](attributes)
        pipe_0_out_split, pipe_0_out_remainder = torch.split(pipe_0_out_full, [self.split_features, self.remainder_features], dim=1)

        pipe_outputs_full = [pipe_0_out_full]
        pipe_outputs_split = [pipe_0_out_split]
        pipe_outputs_remainder = [pipe_0_out_remainder]

        for i in range(1, self.num_pipes):
            pipe_outputs_full.append(self.pipes[i](pipe_outputs_remainder[i-1]))
            if i == (self.num_pipes - 1):
                pipe_outputs_split.append(pipe_outputs_full[i])
            else:
                pipe_outputs_split.append(torch.split(pipe_outputs_full[i], [self.split_features, self.remainder_features], dim=1)[0])
                pipe_outputs_remainder.append(torch.split(pipe_outputs_full[i], [self.split_features, self.remainder_features], dim=1)[1])

        combined_pipes = torch.cat(pipe_outputs_split, dim=1)

        y = self.attention_vector(combined_pipes)

        return x * y, y.cpu().data.numpy().squeeze()
