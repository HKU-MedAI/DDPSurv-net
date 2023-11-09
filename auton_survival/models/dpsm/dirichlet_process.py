import math
import torch
from torch import nn
from torch.distributions import Beta, Categorical, Uniform, MultivariateNormal, Dirichlet
from scipy.stats import beta
import torch.nn.functional as F

from abc import ABCMeta, abstractmethod


class Distribution(metaclass=ABCMeta):
    @abstractmethod
    def __init__(self):
        pass


class ReparametrizedGaussian(Distribution):
    """
    Diagonal ReparametrizedGaussian distribution with parameters mu (mean) and rho. The standard
    deviation is parametrized as sigma = log(1 + exp(rho))
    A sample from the distribution can be obtained by sampling from a unit Gaussian,
    shifting the samples by the mean and scaling by the standard deviation:
    w = mu + log(1 + exp(rho)) * epsilon
    """

    def __init__(self, mu, rho):
        self.mean = mu
        self.rho = rho
        self.normal = torch.distributions.Normal(0, 1)
        self.point_estimate = self.mean

    @property
    def std_dev(self):
        return torch.log1p(torch.exp(self.rho))

    def sample(self, n_samples=1):
        epsilon = torch.distributions.Normal(0, 1).sample(sample_shape=(n_samples, *self.mean.size()))
        epsilon = epsilon.to(self.mean.device)
        return self.mean + self.std_dev * epsilon

    def log_prob(self, target):
        return (-math.log(math.sqrt(2 * math.pi))
                - torch.log(self.std_dev)
                - ((target - self.mean) ** 2) / (2 * self.std_dev ** 2)).sum(dim=-1)

    def entropy(self):
        """
        Computes the entropy of the Diagonal Gaussian distribution.
        Details on the computation can be found in the 'diagonal_gaussian_entropy' notes in the repo
        """
        if self.mean.dim() > 1:
            # n_inputs, n_outputs = self.mean.shape
            dim = 1
            for d in self.mean.shape:
                dim *= d
        elif self.mean.dim() == 0:
            dim = 1
        else:
            dim = len(self.mean)
            # n_outputs = 1

        part1 = dim / 2 * (math.log(2 * math.pi) + 1)
        part2 = torch.sum(torch.log(self.std_dev))

        return (part1 + part2).unsqueeze(0)


class DirichletProcess(nn.Module):
    def __init__(self, concentration, trunc, eta, batch_size, dim=1024, n_sample=100):
        super().__init__()
        self.alpha = concentration
        self.T = trunc
        self.dim = dim
        self.batch_size = batch_size

        self.mu = nn.ParameterList([nn.Parameter(torch.zeros([self.dim])) for t in range(self.T)])
        # self.sig = nn.Parameter(torch.stack([torch.eye(self.dim) for _ in range(self.T)]))
        self.rho = nn.ParameterList([nn.Parameter(torch.zeros([self.dim])) for t in range(self.T)])
        self.gaussians = [ReparametrizedGaussian(self.mu[t], self.rho[t]) for t in range(self.T)]
        self.phi = torch.ones([self.T, self.dim]) / self.T

        self.eta = eta
        self.gamma_1 = torch.ones(self.T)
        self.gamma_2 = torch.ones(self.T) * eta

    def mix_weights(self, beta):
        beta1m_cumprod = (1 - beta).cumprod(-1)
        return F.pad(beta, (0, 1), value=1) * F.pad(beta1m_cumprod, (1, 0), value=1)

    def entropy(self):
        entropy = [self.gaussians[t].entropy() for t in range(self.T)]
        entropy = torch.stack(entropy, dim=0).mean()

        return entropy

    def get_log_prob(self, x):
        pdfs = [self.gaussians[t].log_prob(x) for t in range(self.T)]
        pdfs = torch.stack(pdfs, dim=-1)
        return pdfs

    def sample_beta(self, size):
        a = self.gamma_1.detach().cpu().numpy()
        b = self.gamma_2.detach().cpu().numpy()

        samples = beta.rvs(a, b, size=(size, self.T))
        samples = torch.from_numpy(samples).cuda()

        return samples

    def forward(self, x):
        beta = self.sample_beta(x.shape[1])
        pi = self.mix_weights(beta)[:, :-1]
        entropy = self.entropy()
        log_pdfs = self.get_log_prob(x)

        self.phi = torch.softmax(torch.log(pi) + entropy + log_pdfs, dim=-1)

        likelihood = self.phi * (entropy + log_pdfs)
        likelihood = likelihood.sum(1).mean(0)

        return likelihood

    def inference(self, x):
        """
        Get logit

        return: Logits with length T
        """
        beta = self.sample_beta(x.shape[1])
        pi = self.mix_weights(beta)[:, :-1]
        log_pdfs = self.get_log_prob(x)
        logits = pi * torch.softmax(log_pdfs, dim=2)

        return logits

    def update_gamma(self):

        phi = self.phi

        phi_flipped = torch.flip(phi, dims=[1])
        cum_sum = torch.cumsum(phi_flipped, dim=1) - phi_flipped
        cum_sum = torch.flip(cum_sum, dims=[1])

        self.gamma_1 = 1 + phi.sum(0)
        self.gamma_2 = self.eta + cum_sum.sum(0)

    def update_mean(self, phi):
        N = phi.sum(0)
        pass

    def update_variance(self):
        pass
