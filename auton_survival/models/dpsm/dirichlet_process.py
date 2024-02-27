import math
import torch
from torch import nn
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
        return torch.exp(self.rho)

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

    def set_parameters(self, mu, rho):
        self.mean = mu
        self.rho = rho


class DirichletProcess(nn.Module):
    def __init__(self, trunc, eta, batch_size, dim=1024, n_sample=100):
        super().__init__()
        self.T = trunc
        self.dim = dim
        self.batch_size = batch_size
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"

        self.mu = torch.FloatTensor(self.T, self.dim).uniform_(-0.5, 0.5).to(self.device)
        self.rho = torch.FloatTensor(self.T, self.dim).uniform_(-4, -3).to(self.device)
        # self.mu_encode = nn.Sequential(
        #     nn.Linear(dim, dim * 2),
        #     # nn.Tanh(),
        #     # nn.Linear(dim * 2, dim * 2),
        #     nn.Tanh(),
        #     nn.Linear(dim * 2, dim * trunc)
        # )
        # self.rho_encode = nn.Sequential(
        #     nn.Linear(dim, dim * 2),
        #     # nn.Tanh(),
        #     # nn.Linear(dim * 2, dim * 2),
        #     nn.Tanh(),
        #     nn.Linear(dim * 2, dim * trunc)
        # )
        self.gaussians = [ReparametrizedGaussian(self.mu[t], self.rho[t]) for t in range(self.T)]
        self.phi = (torch.ones([self.batch_size, self.T]) / self.T).to(self.device)

        self.eta = eta
        self.gamma_1 = torch.ones(self.T).cuda()
        self.gamma_2 = (torch.ones(self.T) * eta).cuda()
        self.params = nn.ParameterList([self.phi])

    def sample_beta(self, size):
        a = self.gamma_1.detach().cpu().numpy()
        b = self.gamma_2.detach().cpu().numpy()

        samples = beta.rvs(a, b, size=(size, self.T))
        samples = torch.from_numpy(samples).cuda()

        return samples

    def forward(self, x):
        batch_size = x.shape[-2]

        beta = self.sample_beta(batch_size)
        pi = self.mix_weights(beta)[:, :-1]
        log_pdfs = self.get_log_prob(x)
        entropy = self.entropy()
        entropy = entropy.expand(batch_size, -1)

        phi_new, kl_gaussian = self.get_phi(torch.log(pi), entropy, log_pdfs)
        # log_phi = torch.log(pi) + entropy + log_pdfs
        # phi_new = torch.softmax(log_phi, dim=1)

        self.update_gaussians(x.data, phi_new.data)
        self.update_gamma()

        likelihood = phi_new * kl_gaussian
        likelihood = likelihood.sum(1).mean(0)

        self.phi = phi_new.data

        return - likelihood

    def inference(self, x):
        """
        Get logit

        return: Logits with length T
        """

        beta = self.sample_beta(x.shape[-2])
        pi = self.mix_weights(beta)[:, :-1]

        log_pdfs = self.get_log_prob(x)
        logits = torch.log(pi) + log_pdfs

        # N_t_gaussian = log_pdfs.min()
        # N_t_pi = torch.log(pi).min()
        # mix = (N_t_pi / (N_t_gaussian + N_t_pi))
        # logits = mix * log_pdfs + (1-mix) * torch.log(pi)

        # logits = F.normalize(logits, dim=2)
        logits = F.softmax(logits, dim=-1)

        return logits

    def get_phi(self, log_pi, entropy, log_pdf):
        # TODO: maybe mention this in the paper we do this to improve numerical stability
        kl_gaussian = log_pdf + entropy
        kl_pi = log_pi

        N_t_gaussian = kl_gaussian.min()
        N_t_pi = kl_pi.min()
        mix = (N_t_pi / (N_t_gaussian + N_t_pi))

        kl = mix * kl_gaussian + (1-mix) * kl_pi

        return kl.softmax(dim=1), mix * kl_gaussian
        # return kl.softmax(dim=1),  kl_gaussian

    def update_gamma(self):

        phi = self.phi
        print(phi.shape)
        phi_flipped = torch.flip(phi, dims=[1])
        cum_sum = torch.cumsum(phi_flipped, dim=1) - phi_flipped
        cum_sum = torch.flip(cum_sum, dims=[1])

        self.gamma_1 = 1 + phi.reshape(-1, self.T).mean(0)
        self.gamma_2 = self.eta + cum_sum.reshape(-1, self.T).mean(0)

        # self.gamma_1 = self.gamma_1 + phi.reshape(-1, self.T).mean(0)
        # self.gamma_2 = self.gamma_2 + cum_sum.reshape(-1, self.T).mean(0)

    def mix_weights(self, beta):
        beta1m_cumprod = (1 - beta).cumprod(-1)
        pi = F.pad(beta, (0, 1), value=1) * F.pad(beta1m_cumprod, (1, 0), value=1)
        return pi

    def entropy(self):
        entropy = [self.gaussians[t].entropy() for t in range(self.T)]
        entropy = torch.stack(entropy, dim=-1)

        return entropy

    def get_log_prob(self, x):
        pdfs = [self.gaussians[t].log_prob(x) for t in range(self.T)]
        pdfs = torch.stack(pdfs, dim=-1)
        return pdfs

    def update_mean(self, x, phi_new):
        phi = self.phi
        mu = self.mu

        N_t = phi.sum(0, keepdim=True).clamp(1e-6).T
        N_t_new = phi_new.sum(0, keepdim=True).clamp(1e-6).T

        mu_new = torch.einsum("ij,ik->jk", phi_new.float(), x) / N_t_new
        mix = (N_t / (N_t_new + N_t)).expand(-1, mu.shape[1])
        self.mu = mix * mu + (1 - mix) * mu_new

    def update_variance(self, x, phi_new):
        phi = self.phi
        sig = torch.exp(self.rho)

        # sig_new = torch.einsum("ij,ik->jk", (x - x.mean(0)), (x - x.mean(0))) / x.shape[0]
        # sig_new = torch.einsum("ji,ki->ik", phi_new, sig_new.double())
        # sig_new = torch.diagonal(sig_new).unsqueeze(0).expand(x.shape[0], -1)

        N_t = phi.sum(0, keepdim=True).clamp(1e-6).T
        N_t_new = phi_new.sum(0, keepdim=True).clamp(1e-6).T

        # sig_new = torch.einsum("ij,ik->jk", phi_new.float(), (x - x.mean(0)) ** 2) / N_t_new
        sig_new = (phi_new.unsqueeze(-1) * (x.unsqueeze(1) - self.mu) ** 2).sum(0) / N_t_new
        sig_new = torch.sqrt(sig_new)

        mix = N_t / (N_t_new + N_t)

        updated_sig = mix * sig + (1 - mix) * sig_new
        self.rho = torch.log(updated_sig)

    def update_gaussians(self, x, phi_new):
        self.update_mean(x, phi_new)
        self.update_variance(x, phi_new)
        [self.gaussians[t].set_parameters(self.mu[t], self.rho[t]) for t in range(self.T)]


class DPCluster(nn.Module):
    def __init__(self, trunc, eta, batch_size, epoch=1000, dim=256):
        super().__init__()
        self.T = trunc
        self.dim = dim
        self.batch_size = batch_size
        self.epoch = epoch

        self.eta = eta
        self.gamma_1 = torch.ones(self.T)
        self.gamma_2 = torch.ones(self.T) * eta
        self.dp_process = DirichletProcess(self.T, self.eta, self.batch_size, self.dim)

    def forward(self, x):
        self.dp_process.train()
        optimizer = torch.optim.Adam(self.dp_process.parameters(), lr=5e-3)
        print(self.dp_process.params)
        for i in range(self.epoch):
            optimizer.zero_grad()
            train_loss = self.dp_process(x).mean(0)
            #train_loss.backward()
            optimizer.step()

        return self.dp_process.inference(x)