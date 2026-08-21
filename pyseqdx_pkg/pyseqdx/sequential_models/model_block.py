import torch
import torch.nn as nn

# overall wrapper --------------------------------------------------------------


class SeqNN(nn.Module):
    def __init__(
        self,
        num_t,
        # arguements for RNN/MLP
        input_size,
        hidden_size,
        num_layers=1,
        model="gru",
        dropout=0,
        # pre/post processing prior or after RNN
        pre_proc=None,
        post_proc=None,
        # final touch before output
        link="identity",
        multi_wt = True,
        **kwargs,
    ):
        """Wrapper over core NN blocks for pre/post-processing

        Args:
            num_t (_type_): _description_
            input_size (_type_): _description_
            hidden_size (_type_): _description_
            num_layers (int, optional): _description_. Defaults to 1.
            model (str, optional): What core NN block to use, defaults to "gru".
                - `rnn`, `gru`, `lstm` are RNNs from `torch.nn`;
                - `mlp` is just one MLP applied across all time, one can use time
                conditioning in the pre_proc or time spec block in post_proc to 
                allow variation across time;
                - `series_mlp` is MLP where output at time t is the function of
                prior history till time t.
            dropout (int, optional): _description_. Defaults to 0.
            pre_proc (_type_, optional): _description_. Defaults to None.
            post_proc (_type_, optional): _description_. Defaults to None.
            link (str, optional): _description_. Defaults to "identity".
            
        Attributes:
            logsigma (tensor): size [num_t] for multi-tasks weighting
        """
        super(SeqNN, self).__init__()
        self.num_t = num_t
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.model = model
        # Learnable log variances for multi-tasks weighting
        self.multi_wt = multi_wt
        if multi_wt:
            self.logsigma = nn.Parameter(torch.zeros(num_t))
        else:
            self.logsigma = torch.zeros(num_t)

        if model in ["rnn", "lstm", "gru"]:
            self.nn = SimpleRNN(
                input_size,
                hidden_size,
                num_layers=num_layers,
                model=model,
                dropout=dropout,
            )
        elif model == "mlp":
            self.nn = MLP(
                input_size, hidden_size, output_size=hidden_size, 
                num_layers=num_layers, dropout=dropout
            )
        elif model == "series_mlp":
            self.nn = SeriesMLP(
                num_t, input_size, hidden_size, num_layers=num_layers, dropout=dropout
            )
        elif model == "trans_encoder":
            n_heads = kwargs.get('n_heads', None)
            assert n_heads is not None, "Missing n_heads for trans_encoder"
            self.nn = TransEncoder(
                num_t, input_size, n_heads=n_heads, 
                num_layers=num_layers, d_ffn = hidden_size, dropout=dropout
            )
        elif model == "identity":
            self.nn = nn.Identity()
        else:
            raise ValueError(f"unknown model: {model}")

        # post-processing layer decoder (e.g., timespec)
        if post_proc is None:
            self.post_proc = nn.Identity()
        else:
            self.post_proc = post_proc
        # pre-processing layer encoder (e.g., timespec)
        if pre_proc is None:
            self.pre_proc = nn.Identity()
        else:
            self.pre_proc = pre_proc

        if link in ("identity", "none"):
            self.link = nn.Identity()
        elif link == "relu":
            self.link = nn.ReLU()
        elif link == "sigmoid":
            self.link = nn.Sigmoid()
        elif link == "tanh":
            self.link = nn.Tanh()
        else:
            raise ValueError(f"unknown link: {link}")

    def forward(self, x, return_prelink=False):

        x = self.pre_proc(x)
        out = self.nn(x)
        # out is now [batch, x.size(1), hidden_size]
        out = self.post_proc(out)  # [batch, num_t]

        # use to return the logit
        if not return_prelink:
            out = self.link(out)
        return out  # , hidden_and_cell # well no need for now

# RNN related wrapper ----------------------------------------------------------
# Just to drop hidden state and cell output

class SimpleRNN(nn.Module):
    def __init__(
        self,
        # arguements for RNN/LSTM/GRU
        input_size,
        hidden_size,
        num_layers=1,
        model="gru",
        dropout=0,
        **kwargs,
    ):
        """Wrapper over core RNN blocks for simplification: no hidden state out

        Args:
            input_size (_type_): _description_
            hidden_size (_type_): _description_
            num_layers (int, optional): _description_. Defaults to 1.
            model (str, optional): What core NN block to use, defaults to "gru".
                - `rnn`, `gru`, `lstm` are RNNs from `torch.nn`;
            dropout (int, optional): _description_. Defaults to 0.
        """
        super().__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.model = model

        if model == "rnn":
            self.rnn = nn.RNN(
                input_size,
                hidden_size,
                num_layers=num_layers,
                nonlinearity="relu",
                dropout=dropout,
                batch_first=True,
            )
        elif model == "lstm":
            self.rnn = nn.LSTM(
                input_size,
                hidden_size,
                num_layers=num_layers,
                dropout=dropout,
                batch_first=True,
            )
        elif model == "gru":
            self.rnn = nn.GRU(
                input_size,
                hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout,
            )
        else:
            raise ValueError(f"unknown model: {model}")
        
    def forward(self, x):
        # no hidden state in and out for simplification
        out, hidden_state_cell = self.rnn(x)
        return out

# MLP related ------------------------------------------------------------------


# a simple MLP
class MLP(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        output_size,
        num_layers=1,
        dropout=0,
    ):
        """_summary_

        Args:
            input_size (_type_): _description_
            hidden_size (_type_): _description_
            output_size (_type_): _description_
            num_layers (int, optional): Defaults to 1, i.e., no hidden layer
        """
        super(MLP, self).__init__()

        layers = []
        if num_layers == 1:
            layers.append(nn.Linear(input_size, output_size))
        else:
            # Input layer
            layers.append(nn.Linear(input_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))

            # Hidden layers
            for _ in range(num_layers - 2):
                layers.append(nn.Linear(hidden_size, hidden_size))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(dropout))

            # Output layer
            layers.append(nn.Linear(hidden_size, output_size))

        # Combine all layers into a sequential model
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)


# a series of MLP with causality
class SeriesMLP(nn.Module):
    def __init__(
        self,
        num_t,
        input_size,
        hidden_size,
        num_layers=1,
        dropout=0,
    ):
        """MLP handling sequential data

        Args:
            num_t (_type_): _description_
            input_size (_type_): _description_
            hidden_size (_type_): which is also output size.
            num_layers (int, optional): Defaults to 1, i.e., no hidden layer
            dropout (int, optional): _description_. Defaults to 0.
        """
        super(SeriesMLP, self).__init__()
        self.num_t = num_t
        self.hidden_size = hidden_size
        self.submodels = nn.ModuleList(
            [
                MLP((t + 1) * input_size, hidden_size, hidden_size, num_layers, dropout)
                for t in range(num_t)
            ]
        )
        self.flatten = nn.Flatten(start_dim=-2, end_dim=-1)

    def forward(self, x):
        # x: [batch_size, num_t, input_size]
        # output: [batch_size, num_t, hidden_size]

        out = []
        for t in range(x.size(1)):
            history = x[:, range(t + 1), :]  # [batch_size, t, input_size]
            flat_hist = self.flatten(history)  # [batch_size, (t+1) * input_size]
            output = self.submodels[t](flat_hist)  # [batch_size, hidden_size]
            out.append(output)

        out = torch.stack(out, dim=1)  # [batch_size, num_t, hidden_size]

        return out


# Transformer ------------------------------------------------------------------

# Don't forget to use TimeConditioner in timespec.py for positional encoding.
class TransEncoder(nn.Module):
    # input  [batch_size, num_t, n_features]
    # output [batch_size, num_t, n_features]
    def __init__(
        self,
        num_t: int,  # number of time points
        n_features: int,  # number of dynamic features
        n_heads: int,
        num_layers: int = 1,
        d_ffn: int = 8,  # dimension of feedfowardword network
        dropout: int = 0,
    ):

        super().__init__()

        # encoder layer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=n_features,
            nhead=n_heads,
            dim_feedforward=d_ffn,
            dropout=dropout,
            batch_first=True,
        )  # nheads must be dividable by n_features
        self.trans_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        mask = (torch.triu(torch.ones((num_t, num_t))) == 1).transpose(0, 1)
        mask = (
            mask.float()
            .masked_fill(mask == 0, float("-inf"))
            .masked_fill(mask == 1, float(0.0))
        )
        # mask = torch.tril(torch.ones((num_t, num_t)))
        self.register_buffer("mask", mask)

    def forward(self, x):
        # x shape [batch_size, num_t, n_features]

        # Encoder
        encoder_out = self.trans_encoder(
            x, mask=self.mask, is_causal=True
        )  # [batch_size, num_t, n_features]

        return encoder_out


# TBD: make use of the auto-recursive eval of TransformerDecoderLayer
