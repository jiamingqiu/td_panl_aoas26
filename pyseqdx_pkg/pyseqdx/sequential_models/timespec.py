import torch
import torch.nn as nn
from math import sqrt, log

# time specific and time conditioning

# Linear time specific layers --------------------------------------------------
# input:    [batch_size, num_t, input_size]
# output:   [batch_size, num_t, output_size]
# output[i, t, ] = weight[t] @ input[i, t, ] + bias[t] for all t = 1, ..., num_t


class TimeSpecLinearLayer(nn.Module):
    def __init__(self, num_t, input_size, output_size):
        super().__init__()

        # Use similar initialization as that used in
        # a fully connected linear layer, otherwise performs differently.
        init_bound = 1 / sqrt(input_size)
        # weight tensor of shape [seq_len, input_size, output_size]
        self.weight = nn.Parameter(
            torch.empty(num_t, input_size, output_size).uniform_(
                -init_bound, init_bound
            )
        )

        # bias tensor of shape [seq_len, output_size]
        self.bias = nn.Parameter(
            torch.empty(num_t, output_size).uniform_(-init_bound, init_bound)
        )

    def forward(self, x):
        # x is expected to be of shape [batch, seq_len, input_size]

        # Unsqueeze to shape [batch, seq_len, input_size, 1]
        x = x.unsqueeze(-1)

        # Elementwise product with the weight tensor
        # result shape [batch, seq_len, input_size, output_size]
        x = x * self.weight

        # reduce the shape to [batch, seq_len, output_size]
        x = x.sum(dim=-2)

        # Add the bias tensor, which is broadcasted over the batch dimension
        x = x + self.bias

        return x


class TimeSpecLinearBlock(nn.Module):
    def __init__(
        self,
        num_t,
        input_size,
        hidden_size,
        output_size,
        num_layers,
        dropout=0,
        activation="relu",
        multi_wt=True,
        **kwargs,
    ):
        super().__init__()
        self.num_t = num_t
        self.num_layers = num_layers

        # in case one want some uncertainty weighting
        if multi_wt:
            self.logsigma = nn.Parameter(torch.zeros(num_t))
        else:
            self.logsigma = torch.zeros(num_t)

        if activation is None:
            self.activation = nn.Identity()
        elif activation == "relu":
            self.activation = nn.ReLU()
        else:
            self.activation = activation

        if num_layers == 0: 
            # for compatibility, this means no time-spec layer
            self.mlp = nn.Linear(input_size, output_size)
        elif num_layers == 1:
            self.mlp = TimeSpecLinearLayer(num_t, input_size, output_size)
        else:
            layers = []
            # Input layer
            layers.append(TimeSpecLinearLayer(num_t, input_size, hidden_size))
            layers.append(self.activation)
            layers.append(nn.Dropout(dropout))

            # Hidden layers
            for _ in range(num_layers - 2):
                layers.append(TimeSpecLinearLayer(num_t, hidden_size, hidden_size))
                layers.append(self.activation)
                layers.append(nn.Dropout(dropout))

            # Output layer
            layers.append(TimeSpecLinearLayer(num_t, hidden_size, output_size))

            # Combine all layers into a sequential model
            self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        # x: [batch_size, num_t, input_size]
        # out: [batch_size, num_t, output_size]
        # out[i, t, :] only depends on x[i, t, :]
        out = self.mlp(x)
        return out


# Time conditioner -------------------------------------------------------------


# this will also be used as positional encoding for Transformer
class TimeConditioner(nn.Module):
    def __init__(
        self,
        # time encoder related
        encode_size,
        encode_method,
        encoder=None,
        # how to combine encoding
        conditioning_method="add",
    ):
        """Time conditioning

        Args:
            encode_size (int): dim of time code
            encode_method (str): 'mlp' or 'sinusoidal'
            encoder (optional): User specific time encoder. Defaults to None.
            conditioning_method (str, optional): 'add' or 'concat'. Defaults to 'add'.

        Output:
            If conditioning method is add, then input + time_code,
            otherwise concat, concat([input, time_code], dim = -1).
            So for add, make sure the encode_size is the same as input size.
        """
        super().__init__()
        if encoder is None and conditioning_method != "none":
            self.encoder = TimeEncoder(encode_size, encode_method)
        else:
            self.encoder = nn.Identity() # just a place holder

        if conditioning_method == "add":
            self.conditioner = self.condition_add
        elif conditioning_method == "concat":
            self.conditioner = self.condition_concat
        elif conditioning_method == "none":
            self.conditioner = self.condition_id
        else:
            raise ValueError(f"Unknown conditioning method: {conditioning_method}")

    def condition_add(self, x, code):
        # x size: [batch, num_t, input_size]
        # code size: [num_t, input_size]
        # i.e., input_size must be the same as encode_size
        return x + code

    def condition_concat(self, x, code):
        # x size: [batch, num_t, input_size]
        # code size: [num_t, encode_size]
        # input_size need not be the same as encode_size
        return torch.concat([x, code], dim=-1)

    def condition_id(self, x, code):
        return x

    def forward(self, x):
        # x size: [batch, num_t, input_size]
        time_code = self.encoder(x.size(1))  # size [num_t, encode_size]
        return self.conditioner(x, time_code)


# Time encoder -----------------------------------------------------------------


class TimeEncoder(nn.Module):
    def __init__(self, output_size, method="mlp", hidden_size=16):
        super().__init__()
        method = method.lower()
        if method == "mlp":
            self.encoder = TimeEncoderMLP(output_size, hidden_size)
        elif method == "sinusoidal":
            self.encoder = TimeEncoderSinusoidal(output_size)
        else:
            raise ValueError(f"Unknown encoding method: {method}")

    def forward(self, num_t):
        """
        Args:
            num_t: int, number of time
        Return:
            tensor shape [num_t, output_size]
        """
        return self.encoder(num_t)


class TimeEncoderMLP(nn.Module):
    def __init__(self, output_size, hidden_size=16, max_len=5000):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, hidden_size), nn.ReLU(), nn.Linear(hidden_size, output_size)
        )
        t_seq = 1.0 * torch.arange(max_len).unsqueeze(1)  # size [max_len, 1]
        self.register_buffer("t_seq", t_seq)

    def forward(self, num_t):
        """
        Args:
            num_t: int, number of time
        Return:
            tensor shape [num_t, output_size]
        """
        use_t_seq = self.t_seq[range(num_t)]
        return self.net(use_t_seq)  # [*, output_size]


class TimeEncoderSinusoidal(nn.Module):
    # Modified from https://pytorch.org/tutorials/beginner/transformer_tutorial.html
    def __init__(self, output_size, max_len: int = 5000):
        super().__init__()
        assert output_size % 2 == 0, "Sinusoidal encoding requires even output_size"
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, output_size, 2) * (-log(10000.0) / output_size)
        )
        pe = torch.zeros(max_len, output_size)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, num_t):
        """
        Args:
            num_t: int, number of time
        Return:
            tensor shape [num_t, output_size]
        """
        return self.pe[range(num_t), :]
