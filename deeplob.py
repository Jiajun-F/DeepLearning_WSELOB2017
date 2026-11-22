"""DeepLOB from Zhang, Zohren, and Roberts, Section IV and Figures 3-4.

See https://arxiv.org/abs/1808.03668. This implements the paper's channel counts,
activations, and time-preserving padding, with logits for CrossEntropyLoss.
Optional normalization, dropout, and initialization settings support comparisons.
"""

import torch
from torch import nn


class DeepLOB(nn.Module):
    """Map (batch, time, 4 * n_levels) order book windows to three class logits."""

    def __init__(self, n_levels=10, *, batch_norm=False, dropout=0.0, initialization="pytorch"):
        super().__init__()
        self.n_levels = n_levels

        # Pair each price with its volume. Feature width: 4*n -> 2*n.
        # For each (4, 1) convolution, pad time by one row before and two after.
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=(1, 2), stride=(1, 2)),
            nn.LeakyReLU(0.01),
            nn.ZeroPad2d((0, 0, 1, 2)),
            nn.Conv2d(16, 16, kernel_size=(4, 1)),
            nn.LeakyReLU(0.01),
            nn.ZeroPad2d((0, 0, 1, 2)),
            nn.Conv2d(16, 16, kernel_size=(4, 1)),
            nn.LeakyReLU(0.01),
        )

        # Combine ask and bid information at each level. Width: 2*n -> n.
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=(1, 2), stride=(1, 2)),
            nn.LeakyReLU(0.01),
            nn.ZeroPad2d((0, 0, 1, 2)),
            nn.Conv2d(16, 16, kernel_size=(4, 1)),
            nn.LeakyReLU(0.01),
            nn.ZeroPad2d((0, 0, 1, 2)),
            nn.Conv2d(16, 16, kernel_size=(4, 1)),
            nn.LeakyReLU(0.01),
        )

        # Combine all levels, leaving one feature position per time step.
        self.conv3 = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=(1, n_levels)),
            nn.LeakyReLU(0.01),
            nn.ZeroPad2d((0, 0, 1, 2)),
            nn.Conv2d(16, 16, kernel_size=(4, 1)),
            nn.LeakyReLU(0.01),
            nn.ZeroPad2d((0, 0, 1, 2)),
            nn.Conv2d(16, 16, kernel_size=(4, 1)),
            nn.LeakyReLU(0.01),
        )

        # Three parallel branches capture patterns at different time scales.
        self.inception3 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=(1, 1)),
            nn.LeakyReLU(0.01),
            nn.Conv2d(32, 32, kernel_size=(3, 1), padding=(1, 0)),
            nn.LeakyReLU(0.01),
        )
        self.inception5 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=(1, 1)),
            nn.LeakyReLU(0.01),
            nn.Conv2d(32, 32, kernel_size=(5, 1), padding=(2, 0)),
            nn.LeakyReLU(0.01),
        )
        self.inception_pool = nn.Sequential(
            # The paper specifies zero padding for the pooling branch too.
            nn.ZeroPad2d((0, 0, 1, 1)),
            nn.MaxPool2d(kernel_size=(3, 1), stride=(1, 1)),
            nn.Conv2d(16, 32, kernel_size=(1, 1)),
            nn.LeakyReLU(0.01),
        )

        self.lstm = nn.LSTM(input_size=96, hidden_size=64, batch_first=True)
        self.classifier = nn.Linear(64, 3)

        # Drop complete channels with one mask shared across time, before the LSTM.
        # Dropout is disabled by model.eval(); zero probability preserves the baseline.
        self.dropout = nn.Dropout1d(dropout)
        if batch_norm:
            self._add_batch_norm()
        self._initialize_weights(initialization)

    def _add_batch_norm(self):
        """Place BatchNorm after each convolution activation, as in the PyTorch reference."""
        for name in ("conv1", "conv2", "conv3", "inception3", "inception5", "inception_pool"):
            layers = []
            for layer in getattr(self, name):
                layers.append(layer)
                if isinstance(layer, nn.Conv2d):
                    channels = layer.out_channels
                elif isinstance(layer, nn.LeakyReLU):
                    layers.append(nn.BatchNorm2d(channels))
            setattr(self, name, nn.Sequential(*layers))

    def _initialize_weights(self, method):
        """Initialize convolutions; custom presets use an orthogonal LSTM recurrent matrix."""
        if method == "pytorch":
            return
        if method not in ("xavier", "kaiming", "he_normal"):
            raise ValueError("Choose initialization='pytorch', 'xavier', 'kaiming', or 'he_normal'.")

        for layer in self.modules():
            if isinstance(layer, nn.Conv2d):
                if method == "xavier":
                    nn.init.xavier_uniform_(layer.weight)
                elif method == "kaiming":
                    nn.init.kaiming_uniform_(layer.weight, a=0.01, nonlinearity="leaky_relu")
                else:
                    nn.init.kaiming_normal_(layer.weight, a=0.01, mode="fan_in", nonlinearity="leaky_relu")
                nn.init.zeros_(layer.bias)
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)

        # Keras-style LSTM: Xavier input weights, one orthogonal stacked recurrent matrix,
        # and a total forget-gate bias of one. PyTorch adds its two bias vectors.
        nn.init.xavier_uniform_(self.lstm.weight_ih_l0)
        nn.init.orthogonal_(self.lstm.weight_hh_l0)
        nn.init.zeros_(self.lstm.bias_ih_l0)
        nn.init.zeros_(self.lstm.bias_hh_l0)
        hidden = self.lstm.hidden_size
        with torch.no_grad():
            self.lstm.bias_ih_l0[hidden:2 * hidden].fill_(1)

    def forward(self, x):
        """Return logits for the window's final prediction position."""
        if x.ndim != 3 or x.shape[-1] != 4 * self.n_levels:
            raise ValueError("Expected input shape (batch, time, 4 * n_levels).")

        x = x.unsqueeze(1)  # Add the single input channel for Conv2d.
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = torch.cat(
            (self.inception3(x), self.inception5(x), self.inception_pool(x)),
            dim=1,
        )

        # (batch, 96, time, 1) -> (batch, time, 96).
        x = x.squeeze(-1).transpose(1, 2)
        x = self.dropout(x.transpose(1, 2)).transpose(1, 2)
        x, _ = self.lstm(x)
        # CrossEntropyLoss applies log-softmax internally; return raw logits.
        return self.classifier(x[:, -1, :])
