import torch
import torch.nn as nn
from models.layers import LinearSTDP, Conv2dSTDP, DropoutLIF, LocalLearningSignalGenerationLayer
import models.layers.surrogate_gradients as gradients
import torch.optim as optim
import logging
__all__ = ["cifar_tessvgg_model","dvs_tessvgg_model",
           "cifar100_tessvgg_model", "dvscifar10_tessvgg_model"]


class ConvBlock(torch.nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, pool=None, factors=None, dropout=0.0, pool_size=4,
                 activation=None, wn=False, avoid_wn=False):
        super(ConvBlock, self).__init__()
        if pool == "AVG":
            self.pool_layer = nn.AvgPool2d
            if pool_size == 1:
                self.pool = nn.Identity()
            else:
                self.pool = self.pool_layer(pool_size)
        elif pool == "ADAVG":
            self.pool_layer = nn.AdaptiveAvgPool2d
            self.pool = self.pool_layer(pool_size)
        else:
            self.pool_layer = nn.MaxPool2d
            if pool_size == 1:
                self.pool = nn.Identity()
            else:
                self.pool = self.pool_layer(pool_size)
        self.conv = Conv2dSTDP(n_inputs, n_outputs, kernel_size, 1, 1, bias=False,
                               activation=activation, factors=factors, leak=0, wn=wn, avoid_wn=avoid_wn)
        self.dropout = DropoutLIF(dropout)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, Conv2dSTDP):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
            elif isinstance(m, nn.Linear) or isinstance(m, LinearSTDP):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def reset_state(self):
        self.conv.reset_state()
        self.dropout.reset_state()

    def forward(self, input):
        x = self.conv(input)
        x = self.dropout(x)
        x = self.pool(x)
        return x


class VGGConvModel(nn.Module):
    def __init__(self, n_inputs: int = 3, labels: int = 10, activation=None, acc_activation=None,
                 feedback_mode: str = 'LLS', DFA_size=None, factors=None, pool=None, dropout=0.0, gp=1,
                 cosine_lr=200, lr_conv=0.01, wn=False, avoid_wn=False, loss_function="CE", optimizer="SGD"):
        super(VGGConvModel, self).__init__()
        self.labels = labels
        if pool == "AVG":
            self.pool_layer = nn.AvgPool2d
        else:
            self.pool_layer = nn.MaxPool2d

        # Feedback mode BP, DFA, sDFA
        self.feedback_mode = feedback_mode
        if (feedback_mode == "DFA") or (feedback_mode == "sDFA"):
            self.y = torch.zeros(1, labels)
            self.y.requires_grad = False

        else:
            self.y = None

        block1 = ConvBlock(n_inputs, 64, 3, pool, factors, dropout, pool_size=1,
                           activation=activation, wn=wn, avoid_wn=avoid_wn)
        self.conv_block1 = LocalLearningSignalGenerationLayer(block=block1, lr=lr_conv, n_classes=labels,
                                                              weight_decay=0, optimizer=optimizer, training_mode=feedback_mode,
                                                              waveform="square", hidden_dim=4*4*64, pooling_size=4, lr_scheduler="ReduceLROnPlateau",
                                                              cosine_lr=cosine_lr, patience=5, gamma=0.5, loss_function=loss_function)

        block2 = ConvBlock(64, 128, 3, pool, factors, dropout, pool_size=2,
                           activation=activation, wn=wn, avoid_wn=avoid_wn)
        self.conv_block2 = LocalLearningSignalGenerationLayer(block=block2, lr=lr_conv, n_classes=labels,
                                                              weight_decay=0, optimizer=optimizer,
                                                              training_mode=feedback_mode, waveform="square",
                                                              hidden_dim=4 * 4 * 128, pooling_size=2, lr_scheduler="ReduceLROnPlateau",
                                                              cosine_lr=cosine_lr, patience=5, gamma=0.5, loss_function=loss_function)

        block3 = ConvBlock(128, 256, 3, pool, factors, dropout, pool_size=1,
                           activation=activation, wn=wn, avoid_wn=avoid_wn)
        self.conv_block3 = LocalLearningSignalGenerationLayer(block=block3, lr=lr_conv, n_classes=labels,
                                                              weight_decay=0, optimizer=optimizer,
                                                              training_mode=feedback_mode, waveform="square",
                                                              hidden_dim=4 * 4 * 256, pooling_size=2, lr_scheduler="ReduceLROnPlateau",
                                                              cosine_lr=cosine_lr, patience=5, gamma=0.5,
                                                              loss_function=loss_function)

        block4 = ConvBlock(256, 256, 3, pool, factors, dropout, pool_size=2,
                           activation=activation, wn=wn, avoid_wn=avoid_wn)
        self.conv_block4 = LocalLearningSignalGenerationLayer(block=block4, lr=lr_conv, n_classes=labels,
                                                              weight_decay=0, optimizer=optimizer,
                                                              training_mode=feedback_mode, waveform="square",
                                                              hidden_dim=2*2*256, pooling_size=2, lr_scheduler="ReduceLROnPlateau",
                                                              cosine_lr=cosine_lr, patience=5, gamma=0.5,
                                                              loss_function=loss_function)

        block5 = ConvBlock(256, 512, 3, pool, factors, dropout, pool_size=1,
                           activation=activation, wn=wn, avoid_wn=avoid_wn)
        self.conv_block5 = LocalLearningSignalGenerationLayer(block=block5, lr=lr_conv, n_classes=labels,
                                                              weight_decay=0, optimizer=optimizer,
                                                              training_mode=feedback_mode, waveform="square",
                                                              hidden_dim=2*2*512, pooling_size=2, lr_scheduler="ReduceLROnPlateau",
                                                              cosine_lr=cosine_lr, patience=5, gamma=0.5,
                                                              loss_function=loss_function)

        block6 = ConvBlock(512, 512, 3, pool, factors, dropout, pool_size=2,
                           activation=activation, wn=wn, avoid_wn=avoid_wn)
        self.conv_block6 = LocalLearningSignalGenerationLayer(block=block6, lr=lr_conv, n_classes=labels,
                                                              weight_decay=0, optimizer=optimizer,
                                                              training_mode=feedback_mode, waveform="square",
                                                              hidden_dim=2*2*512, pooling_size=2, lr_scheduler="ReduceLROnPlateau",
                                                              cosine_lr=cosine_lr, patience=5, gamma=0.5,
                                                              loss_function=loss_function)

        block7 = ConvBlock(512, 512, 3, pool, factors, dropout, pool_size=1,
                           activation=activation, wn=wn, avoid_wn=avoid_wn)
        self.conv_block7 = LocalLearningSignalGenerationLayer(block=block7, lr=lr_conv, n_classes=labels,
                                                              weight_decay=0, optimizer=optimizer,
                                                              training_mode=feedback_mode, waveform="square",
                                                              hidden_dim=2*2*512, pooling_size=2, lr_scheduler="ReduceLROnPlateau",
                                                              cosine_lr=cosine_lr, patience=5, gamma=0.5,
                                                              loss_function=loss_function)

        block8 = ConvBlock(512, 512, 3, "ADAVG", factors, dropout, pool_size=gp,
                           activation=activation, wn=wn, avoid_wn=avoid_wn)
        self.conv_block8 = LocalLearningSignalGenerationLayer(block=block8, lr=lr_conv, n_classes=labels,
                                                              weight_decay=0, optimizer=optimizer,
                                                              training_mode=feedback_mode,
                                                              waveform="square",
                                                              hidden_dim=512, pooling_size=1, lr_scheduler="ReduceLROnPlateau",
                                                              cosine_lr=cosine_lr, patience=5, gamma=0.5, loss_function=loss_function)


        self.linear = nn.Linear(512*gp*gp, labels, bias=True)
        self._initialize_weights()

    def scheduler_step(self, loss_avg):
        if hasattr(self.conv_block1, "lr_scheduler"):
            if isinstance(self.conv_block1.lr_scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                self.conv_block1.lr_scheduler.step(loss_avg)
                self.conv_block2.lr_scheduler.step(loss_avg)
                self.conv_block3.lr_scheduler.step(loss_avg)
                self.conv_block4.lr_scheduler.step(loss_avg)
                self.conv_block5.lr_scheduler.step(loss_avg)
                self.conv_block6.lr_scheduler.step(loss_avg)
                self.conv_block7.lr_scheduler.step(loss_avg)
                self.conv_block8.lr_scheduler.step(loss_avg)
            else:
                self.conv_block1.lr_scheduler.step()
                self.conv_block2.lr_scheduler.step()
                self.conv_block3.lr_scheduler.step()
                self.conv_block4.lr_scheduler.step()
                self.conv_block5.lr_scheduler.step()
                self.conv_block6.lr_scheduler.step()
                self.conv_block7.lr_scheduler.step()
                self.conv_block8.lr_scheduler.step()


    def optimizer_zero_grad(self):
        self.conv_block1.optimizer_zero_grad()
        self.conv_block2.optimizer_zero_grad()
        self.conv_block3.optimizer_zero_grad()
        self.conv_block4.optimizer_zero_grad()
        self.conv_block5.optimizer_zero_grad()
        self.conv_block6.optimizer_zero_grad()
        self.conv_block7.optimizer_zero_grad()
        self.conv_block8.optimizer_zero_grad()

    def optimizer_step(self):
        self.conv_block1.optimizer_step()
        self.conv_block2.optimizer_step()
        self.conv_block3.optimizer_step()
        self.conv_block4.optimizer_step()
        self.conv_block5.optimizer_step()
        self.conv_block6.optimizer_step()
        self.conv_block7.optimizer_step()
        self.conv_block8.optimizer_step()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, Conv2dSTDP):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
            elif isinstance(m, nn.Linear) or isinstance(m, LinearSTDP):
                nn.init.kaiming_normal_(m.weight)
                # m.weight.data = 0.5 * (torch.rand_like(m.weight.data) - 0.5)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def reset_states(self):
        self.conv_block1.block.reset_state()
        self.conv_block2.block.reset_state()
        self.conv_block3.block.reset_state()
        self.conv_block4.block.reset_state()
        self.conv_block5.block.reset_state()
        self.conv_block6.block.reset_state()
        self.conv_block7.block.reset_state()
        self.conv_block8.block.reset_state()

    def update_batch_size(self, x:torch.Tensor):
        if (self.feedback_mode == "DFA") or (self.feedback_mode == "sDFA"):
            self.y = torch.zeros(x.shape[0], self.labels, device=x.device)
            self.y.requires_grad = False
        else:
            self.y = None

    def forward(self, x, target=None):
        self.update_batch_size(x)

        training = self.training
        batch_size = x.shape[0]
        x = self.conv_block1(x, target)
        x = self.conv_block2(x, target)
        x = self.conv_block3(x, target)
        x = self.conv_block4(x, target)
        x = self.conv_block5(x, target)
        x = self.conv_block6(x, target)
        x = self.conv_block7(x, target)
        x = self.conv_block8(x, target)
        x = x.view(batch_size, -1)
        x = self.linear(x)
        return x


def dvs_tessvgg_model(args, device):
    if args.activation != "LinearSpike":
        activation = gradients.__dict__[args.activation]
        logging.info("Activation used: "+args.activation)
    else:
        activation = None
        logging.info("Activation used: None")
    acc_act = None

    factors = args.factors_stdp

    model = VGGConvModel(n_inputs=2, labels=11, activation=activation, acc_activation=acc_act,
                       factors=factors, pool=args.pooling, cosine_lr=args.scheduler, lr_conv=args.lr_conv, wn=args.wn,
                       avoid_wn=args.avoid_wn, loss_function=args.loss, optimizer=args.optimizer)

    if args.pretrained_model:
        model.load_state_dict(torch.load(args.pretrained_model)['state_dict'], strict=False)
    return model


def dvscifar10_tessvgg_model(args, device):
    if args.activation != "LinearSpike":
        activation = gradients.__dict__[args.activation]
        logging.info("Activation used: "+args.activation)
    else:
        activation = None
        logging.info("Activation used: None")
    acc_act = None

    factors = args.factors_stdp

    model = VGGConvModel(n_inputs=2, labels=10, activation=activation, acc_activation=acc_act,
                       factors=factors, pool=args.pooling, cosine_lr=args.scheduler, lr_conv=args.lr_conv, wn=args.wn,
                       avoid_wn=args.avoid_wn, loss_function=args.loss, optimizer=args.optimizer)

    if args.pretrained_model:
        model.load_state_dict(torch.load(args.pretrained_model)['state_dict'], strict=False)
    return model


def cifar_tessvgg_model(args, device):
    if args.activation != "LinearSpike":
        activation = gradients.__dict__[args.activation]
        logging.info("Activation used: "+args.activation)
    else:
        activation = None
        logging.info("Activation used: None")
    acc_act = None

    factors = args.factors_stdp

    model = VGGConvModel(n_inputs=3, labels=10, activation=activation, acc_activation=acc_act,
                       factors=factors, pool=args.pooling, cosine_lr=args.scheduler, lr_conv=args.lr_conv, wn=args.wn,
                       avoid_wn=args.avoid_wn, loss_function=args.loss, optimizer=args.optimizer)

    if args.pretrained_model:
        model.load_state_dict(torch.load(args.pretrained_model)['state_dict'], strict=False)
    return model


def cifar100_tessvgg_model(args, device):
    if args.activation != "LinearSpike":
        activation = gradients.__dict__[args.activation]
        logging.info("Activation used: "+args.activation)
    else:
        activation = None
        logging.info("Activation used: None")
    acc_act = None

    factors = args.factors_stdp

    model = VGGConvModel(n_inputs=3, labels=100, activation=activation, acc_activation=acc_act,
                       factors=factors, pool=args.pooling, cosine_lr=args.scheduler, lr_conv=args.lr_conv, wn=args.wn,
                       avoid_wn=args.avoid_wn, loss_function=args.loss, optimizer=args.optimizer)

    if args.pretrained_model:
        model.load_state_dict(torch.load(args.pretrained_model)['state_dict'], strict=False)
    return model

