import torch.nn as nn
import torch

device = torch.device('cuda')


class GP_loss_fun(torch.nn.Module):
    def __init__(self, lossfn_type):
        super(GP_loss_fun, self).__init__()
        lossfn_type = lossfn_type.lower()
        while lossfn_type.find(' ') >= 0:
            lossfn_type = lossfn_type.replace(' ', '')
        lossfn_type = lossfn_type.split('+')
        self.loss = []
        self.weight = []
        for this_loss in lossfn_type:
            if this_loss.find('*') >= 0:
                self.weight.append(float(this_loss[:this_loss.find('*')]))
                this_loss = this_loss[this_loss.find('*') + 1:].lower()
            else:
                self.weight.append(1.0)
                this_loss = this_loss.lower()

            if False:
                pass

            elif this_loss in ['mse', 'l2']:
                self.loss.append(nn.MSELoss().to(device))

            else:
                raise NotImplementedError('Loss type not found')

    def forward(self, x, y):
        # x = infer, y = target
        loss = 0
        for idx in range(len(self.loss)):
            loss += self.weight[idx] * self.loss[idx](x, y)
        return loss

