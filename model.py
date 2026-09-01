import torch
import torch.nn as nn


def flatten(tensors):
    return torch.cat([t.reshape(-1) for t in tensors])


def unflatten(v, model):
    idx, out = 0, {}

    for name, p in [
        ("fc1.weight", model.fc1.weight),
        ("fc1.bias", model.fc1.bias),
        ("fc2.weight", model.fc2.weight),
    ]:
        n = p.numel()
        if p.requires_grad or v.numel() == sum(q.numel() for q in model.parameters()):
            out[name] = v[idx : idx + n].view_as(p)
            idx += n
        else:
            out[name] = torch.zeros_like(p)

    return out


class NN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, beta, freeze_first_layer=False):
        super().__init__()
        self.beta = beta
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.freeze_first_layer = freeze_first_layer

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.activation = nn.Tanh()
        self.fc2 = nn.Linear(hidden_dim, output_dim, bias=False)

        self._initialize_weights()

        # Optionally freeze the first layer to recover the random-feature
        # setting in which only fc2 is trained. By default fc1 is trainable,
        # so the model is genuinely nonlinear in its trainable parameters.
        if freeze_first_layer:
            self.fc1.weight.requires_grad = False
            self.fc1.bias.requires_grad = False

    def _initialize_weights(self):
        nn.init.normal_(self.fc1.weight, mean=0.0, std=1.0)
        nn.init.normal_(self.fc1.bias, mean=0.0, std=1.0)
        nn.init.uniform_(self.fc2.weight, a=-1.0, b=1.0)

    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]

    def forward(self, x):
        out = self.activation(self.fc1(x))
        out = self.fc2(out)
        out = (1 / (self.hidden_dim ** self.beta)) * out
        return out


def compute_loss(u, y):
    return torch.sum((u - y) ** 2) / (2 * y.size(0))
