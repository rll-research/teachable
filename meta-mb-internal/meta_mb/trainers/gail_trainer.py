import torch.nn as nn
import torch

# Code found here: https://github.com/Khrylx/PyTorch-RL/tree/master
from babyai.rl.utils.dictlist import merge_dictlists


class Discriminator(nn.Module):
    def __init__(self, num_inputs, hidden_size=(128, 128), activation='tanh'):
        super().__init__()
        if activation == 'tanh':
            self.activation = torch.tanh
        elif activation == 'relu':
            self.activation = torch.relu
        elif activation == 'sigmoid':
            self.activation = torch.sigmoid

        self.affine_layers = nn.ModuleList()
        last_dim = num_inputs
        for nh in hidden_size:
            self.affine_layers.append(nn.Linear(last_dim, nh))
            last_dim = nh

        self.logic = nn.Linear(last_dim, 1)
        self.logic.weight.data.mul_(0.1)
        self.logic.bias.data.mul_(0.0)

    def forward(self, x):
        for affine in self.affine_layers:
            x = self.activation(affine(x))

        prob = torch.sigmoid(self.logic(x))
        return prob


class GailTrainer:
    def __init__(self, args, obs_shape):
        self.args = args
        self.discriminator = Discriminator(obs_shape).to(self.args.device)
        self.optimizer = torch.optim.Adam(self.discriminator.parameters(),
                         self.args.lr, eps=self.args.optim_eps)


    def update(self, expert_batch, policy_batch, train=True):
        combined_batch = merge_dictlists([expert_batch, policy_batch])
        input = torch.cat([combined_batch.obs['obs'], combined_batch.actions], dim=1)
        output = self.discriminator(input)
        labels = torch.cat(torch.zeros(len(expert_batch)), torch.ones(len(policy_batch))).to(self.args.device)
        discrim_loss = nn.BCELoss(output, labels)
        if train:
            self.optimizer.zero_grad()
            discrim_loss.backward()
            self.optimizer.step()
        return {
            'loss': discrim_loss
        }
