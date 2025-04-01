import torch
from torch import nn
import torch.optim as optim
from torch.nn import functional as F

__all__ = ["LocalLearningSignalGenerationLayer"]


class LinearSigmoid(torch.autograd.Function):
    """
    Surrogate gradient based on arctan, used in Feng et al. (2021)
    """
    @staticmethod
    def forward(ctx, x):
        result = torch.zeros_like(x)
        # Segment 1: x <= -2 -> approximate with 0
        result = torch.where(x <= -2, torch.zeros_like(x), result)
        # Segment 2: -2 < x < 2 -> approximate with 0.25 * x + 0.5
        result = torch.where((x > -2) & (x < 2), 0.25 * x + 0.5, result)
        # Segment 3: x >= 2 -> approximate with 1
        result = torch.where(x >= 2, torch.ones_like(x), result)
        return result

    @staticmethod
    def backward(ctx, grad_output):

        return grad_output, None


def generate_frequency_matrix(num_rows, num_cols, min_freq=50, max_freq=2000, freq=None):
    if freq is None:
        frequencies = torch.linspace(min_freq, max_freq, num_rows).unsqueeze(1).cuda()
    else:
        frequencies = freq
    # phases = torch.randn(num_rows, 1) * 2 * 3.14159
    t = torch.arange(num_cols).float().unsqueeze(0).cuda()
    sinusoids = torch.sin(frequencies * t )
    return sinusoids


def compute_LLS(activation, labels, temperature=1, label_smoothing=0.0, act_size=1, n_classes=10,
                modulation_term=None, modulation=False, freq=None, waveform="cosine", loss_function="CE"):
    batch_size = activation.size(0)
    if activation.dim() == 4:
        latents = F.adaptive_avg_pool2d(activation, (act_size, act_size)).view(batch_size, -1)
    else:
        latents = F.adaptive_avg_pool1d(activation, act_size).view(batch_size, -1)
    basis = generate_frequency_matrix(n_classes, latents.size(1), max_freq=512, freq=freq).cuda()
    # basis = generate_frequency_matrix(n_classes, latents.size(1), max_freq=latents.size(1) - 50).cuda()
    if waveform == "square":
        basis = torch.sign(basis)
    basis = basis/latents.size(1)
    # latents = F.normalize(latents, dim=1)
    layer_pred = torch.matmul(latents, basis.T)
    if modulation == 1:
        layer_pred = modulation_term*layer_pred
    if modulation == 2:
        layer_pred = torch.matmul(layer_pred, modulation_term)

    if loss_function == "CE":
        loss = torch.nn.functional.cross_entropy(layer_pred / temperature, labels, label_smoothing=label_smoothing)
    elif loss_function == "MSEHW":
        loss = torch.nn.functional.mse_loss(LinearSigmoid.apply(layer_pred / temperature),
                                            torch.nn.functional.one_hot(labels, num_classes=n_classes).float())
    else:
        raise NotImplementedError(f"{loss_function} is not implemented")
    return loss


class LocalLearningSignalGenerationLayer(nn.Module):
    def __init__(self, block:nn.Module, lr=1e-1, n_classes=10, momentum=0, weight_decay=0,
                 nesterov=False, optimizer="SGD", milestones=[10, 30, 50], gamma=0.1, training_mode="LLS",
                 lr_scheduler = "MultiStepLR", patience=20, temperature=1, label_smoothing=0.0, dropout=0.0,
                 waveform="cosine", hidden_dim = 2048, reduced_set=20, pooling_size = 4, scaler = False,
                 cosine_lr=200, loss_function="CE"):
        super(LocalLearningSignalGenerationLayer, self).__init__()
        self.block = block
        self.lr = lr
        self.n_classes = n_classes
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.nesterov = nesterov
        self.training_mode = training_mode
        self.patience = patience
        self.temperature = temperature
        self.label_smoothing = label_smoothing
        self.dropout = dropout
        self.waveform = waveform
        self.milestones = milestones
        self.gamma = gamma
        self.hidden_dim = hidden_dim
        self.reduced_set = reduced_set
        self.pooling_size = pooling_size
        self.scaler = None
        self.loss_function = loss_function

        if optimizer == "SGD":
            self.optimizer = optim.SGD(self.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay,
                                       nesterov=nesterov)
        elif optimizer == "Adam":
            self.optimizer = optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
        else:
            raise ValueError(f"{optimizer} is not supported")

        if lr_scheduler == "MultiStepLR":
            self.lr_scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, gamma=gamma, milestones=milestones)
        elif lr_scheduler == "ReduceLROnPlateau":
            self.lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, factor=gamma, patience=patience)
        elif lr_scheduler == "CosineLR":
            self.lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, cosine_lr)

        self.loss_hist = 0
        self.samples = 0
        self.loss_avg = 0

    def record_statistics(self, loss, batch_size):
        self.loss_hist += loss.item() * batch_size
        self.samples += batch_size
        self.loss_avg = self.loss_hist / self.samples if self.samples > 0 else 0

    def reset_statistics(self):
        self.loss_hist = 0
        self.samples = 0
        self.loss_avg = 0

    def optimizer_zero_grad(self):
        if hasattr(self, "optimizer"):
            self.optimizer.zero_grad()

    def optimizer_step(self):
        if hasattr(self, "optimizer"):
            self.optimizer.step()

    def forward(self, x, labels=None, feedback=None, x_err=None):
        training = self.training

        if self.training_mode == "BP" or not training or labels is None:
            return self.block(x)
        else:
            out = self.block(x.detach())
            if self.training_mode == "LLS":
                temperature = self.temperature
                label_smoothing = self.label_smoothing
                loss = compute_LLS(out, labels, temperature, label_smoothing, self.pooling_size,
                                   self.n_classes, waveform=self.waveform, loss_function=self.loss_function)

            else:
                raise NotImplementedError(f"Unknown training mode: {self.training_mode}")

            # self.optimizer.zero_grad()
            loss.backward()
            # self.optimizer.step()
            self.record_statistics(loss.detach(), x.size(0))

            return out.detach()