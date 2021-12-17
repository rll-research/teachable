import numpy as np
import torch
from torch import nn
import os
import random
from torch import distributions
import math
from torch import distributions as pyd
import torch.nn.functional as F
from torch.distributions import Normal, Categorical


def soft_update_params(net, target_net, tau):
    for param, target_param in zip(net.parameters(), target_net.parameters()):
        target_param.data.copy_(tau * param.data +
                                (1 - tau) * target_param.data)


def set_seed_everywhere(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def make_dir(*path_parts):
    dir_path = os.path.join(*path_parts)
    try:
        os.mkdir(dir_path)
    except OSError:
        pass
    return dir_path


def weight_init(m):
    """Custom weight init for Conv2D and Linear layers."""
    if isinstance(m, nn.Linear):
        m.weight.data.normal_(0, 1)
        m.weight.data *= 1 / torch.sqrt(m.weight.data.pow(2).sum(1, keepdim=True))
        if m.bias is not None:
            m.bias.data.fill_(0)


class MLP(nn.Module):
    def __init__(self,
                 input_dim,
                 hidden_dim,
                 output_dim,
                 hidden_depth,
                 output_mod=None):
        super().__init__()
        self.trunk = mlp(input_dim, hidden_dim, output_dim, hidden_depth,
                         output_mod)
        self.apply(weight_init)

    def forward(self, x):
        return self.trunk(x)


def mlp(input_dim, hidden_dim, output_dim, hidden_depth, output_mod=None):
    if hidden_depth == 0:
        mods = [nn.Linear(input_dim, output_dim)]
    else:
        mods = [nn.Linear(input_dim, hidden_dim), nn.Tanh()]
        for i in range(hidden_depth - 1):
            mods += [nn.Linear(hidden_dim, hidden_dim), nn.Tanh()]
        mods.append(nn.Linear(hidden_dim, output_dim))
    if output_mod is not None:
        mods.append(output_mod)
    trunk = nn.Sequential(*mods)
    return trunk


def to_np(t):
    if t is None:
        return None
    elif t.nelement() == 0:
        return np.array([])
    else:
        return t.cpu().detach().numpy()

# https://medium.com/@kengz/soft-actor-critic-for-continuous-and-discrete-actions-eeff6f651954
class GumbelSoftmax(distributions.RelaxedOneHotCategorical):
    '''
    A differentiable Categorical distribution using reparametrization trick with Gumbel-Softmax
    Explanation http://amid.fish/assets/gumbel.html
    NOTE: use this in place PyTorch's RelaxedOneHotCategorical distribution since its log_prob is not working right (returns positive values)
    Papers:
    [1] The Concrete Distribution: A Continuous Relaxation of Discrete Random Variables (Maddison et al, 2017)
    [2] Categorical Reparametrization with Gumbel-Softmax (Jang et al, 2017)
    '''

    def sample(self, sample_shape=torch.Size(), one_hot=False):
        '''Gumbel-softmax sampling. Note rsample is inherited from RelaxedOneHotCategorical'''
        u = torch.empty(self.logits.size(), device=self.logits.device, dtype=self.logits.dtype).uniform_(0, 1)
        noisy_logits = self.logits - torch.log(-torch.log(u))
        argmax = torch.argmax(noisy_logits, dim=-1, keepdim=True)
        if one_hot:
            argmax = F.one_hot(argmax[:, 0].long(), self.logits.shape[-1]).float()
        return argmax

    def rsample(self, sample_shape=torch.Size(), one_hot=False):
        soft_sample = super().rsample(sample_shape)
        hard_sample = torch.argmax(soft_sample, dim=-1, keepdim=True)
        if one_hot:
            hard_sample = F.one_hot(hard_sample[:, 0].long(), self.logits.shape[-1]).float()
        return soft_sample, hard_sample

    def log_prob(self, value):
        '''value is one-hot or relaxed'''
        if value.shape != self.logits.shape:
            value = F.one_hot(value[:, 0].long(), self.logits.shape[-1]).float()
            assert value.shape == self.logits.shape
        return - torch.sum(- value * F.log_softmax(self.logits, -1), -1)

# Inspired by FiLMedBlock from https://arxiv.org/abs/1709.07871
class FiLM(nn.Module):
    def __init__(self, in_features, out_features, in_channels, imm_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels=in_channels, out_channels=imm_channels,
            kernel_size=(3, 3), padding=1)
        self.bn1 = nn.BatchNorm2d(imm_channels)
        self.conv2 = nn.Conv2d(
            in_channels=imm_channels, out_channels=out_features,
            kernel_size=(3, 3), padding=1)
        self.bn2 = nn.BatchNorm2d(out_features)

        self.weight = nn.Linear(in_features, out_features)
        self.bias = nn.Linear(in_features, out_features)

        self.apply(weight_init)

    def forward(self, x, y):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.conv2(x)
        weight = self.weight(y).unsqueeze(2).unsqueeze(3)
        bias = self.bias(y).unsqueeze(2).unsqueeze(3)
        out = x * weight + bias
        return F.relu(self.bn2(out))


class ImageBOWEmbedding(nn.Module):
    def __init__(self, max_value, embedding_dim):
        super().__init__()
        self.max_value = max_value
        self.embedding_dim = embedding_dim
        self.embedding = nn.Embedding(3 * max_value, embedding_dim)
        self.apply(weight_init)

    def forward(self, inputs):
        offsets = torch.Tensor([0, self.max_value, 2 * self.max_value]).to(inputs.device)
        inputs = (inputs + offsets[None, :, None, None]).long()
        return self.embedding(inputs).sum(1).permute(0, 3, 1, 2)


class TanhTransform(pyd.transforms.Transform):
    domain = pyd.constraints.real
    codomain = pyd.constraints.interval(-1.0, 1.0)
    bijective = True
    sign = +1

    def __init__(self, cache_size=1):
        super().__init__(cache_size=cache_size)

    @staticmethod
    def atanh(x):
        return 0.5 * (x.log1p() - (-x).log1p())

    def __eq__(self, other):
        return isinstance(other, TanhTransform)

    def _call(self, x):
        return x.tanh()

    def _inverse(self, y):
        # We do not clamp to the boundary here as it may degrade the performance of certain algorithms.
        # one should use `cache_size=1` instead
        return self.atanh(y)

    def log_abs_det_jacobian(self, x, y):
        # We use a formula that is more numerically stable, see details in the following link
        # https://github.com/tensorflow/probability/commit/ef6bb176e0ebd1cf6e25c6b5cecdd2428c22963f#diff-e120f70e92e6741bca649f04fcd907b7
        return 2. * (math.log(2.) - x - F.softplus(-2. * x))


class SquashedNormal(pyd.transformed_distribution.TransformedDistribution):
    def __init__(self, loc, scale):
        self.loc = loc
        self.scale = scale

        self.base_dist = pyd.Normal(loc, scale)
        transforms = [TanhTransform()]
        super().__init__(self.base_dist, transforms)

    @property
    def mean(self):
        mu = self.loc
        for tr in self.transforms:
            mu = tr(mu)
        return mu

    def log_prob(self, value):
        eps = 1e-3
        value = torch.clamp(value, -1 + eps, 1 - eps)
        return super().log_prob(value)


class DiagGaussianActor(nn.Module):
    """torch.distributions implementation of an diagonal Gaussian policy."""

    def __init__(self, obs_dim, action_dim, hidden_dim=1024, hidden_depth=2, log_std_bounds=(-20, 2), discrete=False):
        super().__init__()

        self.discrete = discrete
        self.log_std_bounds = log_std_bounds
        if not discrete:
            action_dim *= 2
        self.trunk = mlp(obs_dim, hidden_dim, action_dim,
                               hidden_depth)

        self.outputs = dict()
        self.apply(weight_init)

    def forward(self, obs):
        if self.discrete:
            logits = self.trunk(obs)
            dist = Categorical(logits=F.log_softmax(logits, dim=1))
            # TODO: consider replacing with gumbel
            # dist = GumbelSoftmax(temperature=1, logits=logits)
        else:
            mu, log_std = self.trunk(obs).chunk(2, dim=-1)

            # constrain log_std inside [log_std_min, log_std_max]
            log_std_min, log_std_max = self.log_std_bounds
            log_std = torch.clamp(log_std, log_std_min, log_std_max)

            std = log_std.exp()

            self.outputs['mu'] = mu
            self.outputs['std'] = std

            dist = Normal(mu, std)
        return dist


class DoubleQCritic(nn.Module):
    """Critic network, employes double Q-learning."""

    def __init__(self, obs_dim, action_dim, hidden_dim=1024, hidden_depth=2, repeat_action=1):
        super().__init__()
        self.repeat_action = repeat_action

        self.Q1 = mlp(obs_dim + action_dim * repeat_action, hidden_dim, 1, hidden_depth)
        self.Q2 = mlp(obs_dim + action_dim * repeat_action, hidden_dim, 1, hidden_depth)

        self.outputs = dict()
        self.apply(weight_init)

    def forward(self, obs, action):
        assert obs.size(0) == action.size(0)

        obs_action = torch.cat([obs] + [action] * self.repeat_action, dim=-1)
        q1 = self.Q1(obs_action)
        q2 = self.Q2(obs_action)

        self.outputs['q1'] = q1
        self.outputs['q2'] = q2

        return q1, q2


class InstrEmbedding(nn.Module):
    def __init__(self, args, env, num_modules=2, image_dim=128, instr_dim=128):
        # Define instruction embedding
        super().__init__()
        self.word_embedding = nn.Embedding(len(env.vocab()), instr_dim)
        self.instr_rnn = nn.GRU(instr_dim, instr_dim, batch_first=True, bidirectional=False)

        self.controllers = []
        for i in range(num_modules):
            mod = FiLM(
                in_features=instr_dim,
                out_features=128 if i < num_modules - 1 else image_dim,
                in_channels=128, imm_channels=128)
            self.controllers.append(mod)
            self.add_module('FiLM_' + str(i), mod)
        self.apply(weight_init)

    def forward(self, obs):
        instruction_vector = obs.instr.long()
        instr_embedding = self._get_instr_embedding(instruction_vector)
        obs.instr = instr_embedding
        x = obs.obs
        for controller in self.controllers:
            x = controller(x, instr_embedding) + x
        x = F.relu(x)
        obs.obs = x
        return obs

    def _get_instr_embedding(self, instr):
        lengths = (instr != 0).sum(1).long()
        out, _ = self.instr_rnn(self.word_embedding(instr))
        hidden = out[range(len(lengths)), lengths - 1, :]
        return hidden


class ImageEmbedding(nn.Module):
    def __init__(self):
        super().__init__()
        self.image_conv = nn.Sequential(*[
            ImageBOWEmbedding(147, 128),
            nn.Conv2d(
                in_channels=128, out_channels=32,
                kernel_size=(8, 8), stride=8, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2),
            nn.Conv2d(in_channels=32, out_channels=128, kernel_size=(3, 3), padding=1, stride=2),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2),
        ])

        # Initialize parameters correctly
        self.apply(weight_init)

    def forward(self, obs):
        inputs = torch.transpose(torch.transpose(obs.obs, 1, 3), 2, 3)
        obs.obs = self.image_conv(inputs)
        return obs

