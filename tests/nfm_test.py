# %%
import numpy as np
import sys
from torch.utils.data import Dataset, DataLoader

sys.path.append('..')  # noqa


import torch
import torch.nn as nn
from nfm.nfm.datasets import SurvivalDataset
from nfm.nfm.base import FullyNeuralNLL
from nfm.nfm.eps_config import ParetoEps
from pycox.evaluation.eval_surv import EvalSurv
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'


class Net(nn.Module):

    def __init__(self, num_features):
        super(Net, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_features=1 + num_features,
                      out_features=128, bias=False),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=1, bias=False)
        )

    def forward(self, y, z):
        inputs = torch.cat([z, y], dim=1)
        # import ipdb
        # ipdb.set_trace()
        return torch.exp(self.mlp(inputs))

# %%


class dataset(Dataset):
    def __init__(self, x_path, t_path, e_path):
        self.x = np.load(x_path)
        self.t = np.load(t_path)
        self.e = np.load(e_path)

    def __getitem__(self, index):
        x_i = self.x[index]
        t_i = self.t[index]
        e_i = self.e[index]
        return x_i, t_i, e_i

    def __len__(self):
        return len(self.x)


# %%
trainset = dataset(x_path='x_train.npy',
                   t_path='t_train.npy', e_path='e_train.npy')
testset = dataset(x_path='x_test.npy',
                  t_path='t_test.npy', e_path='e_test.npy')

# %%
num_features = trainset.x.shape[-1]
num_features

# %%
loader = DataLoader(trainset, batch_size=128)

# %%
x_test, t_test, e_test = testset[:]

# %%
t_test

# %%
n = 0
nll = FullyNeuralNLL(eps_conf=ParetoEps(learnable=True),
                     encoder=Net(num_features=trainset.x.shape[-1])).cuda()
optimizer = torch.optim.Adam(
    lr=1e-3, weight_decay=1e-3, params=nll.parameters())
for i, (x, t, e) in enumerate(loader):
    x = x.to(torch.float32)
    t = torch.unsqueeze(t.to(torch.float32), 1)
    e = e.to(torch.float32)
    nll.train()
    loss = nll(z=x.mean(dim=1).cuda(), y=t.cuda() / 24 / 30, delta=e.cuda())
    print(loss)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    n = n + 1
nll.eval()

# %%
# paras = nll.parameters()
# for para in paras:
#     print(para)

# %%
x_test, t_test, e_test = testset[:]

# %%
t_test.shape

# %%
with torch.no_grad():
    # y_valid, delta_valid, z_valid = valid_folds[i].sort()
    # y_test, delta_test, z_test = test_folds[i].sort()
    # y_valid, y_test = normalize(y_valid), normalize(y_test)
    # valid_loss = nll(z_valid, y_valid, delta_valid)
    # print(z_valid, y_valid, delta_valid)
    # valid_losses.append(valid_loss.item())
    # tg_test = np.linspace(y_test.cpu().numpy().min(), y_test.cpu().numpy().max(), 100)

    x_test, t_test, e_test = testset[:]
    x_train, t_train, e_train = trainset[:]
    horizons = [0.25, 0.5, 0.75, 0.9]

    x = np.concatenate((np.array(x_train), np.array(x_test)), axis=0)
    t = np.concatenate((np.array(t_train), np.array(t_test)), axis=0)
    e = np.concatenate((np.array(e_train), np.array(e_test)), axis=0)

    tg_test = np.quantile(t[e == 1], horizons)

    out_survival = nll.get_survival_prediction(
        z_test=torch.tensor(x_test.mean(axis=1), dtype=torch.float).cuda(), y_test=torch.tensor(tg_test, dtype=torch.float).view(-1, 1).cuda()).cpu().numpy()

    out_risk = 1 - out_survival

    x_train = np.array(x_train)
    t_train = np.array(t_train)
    e_train = np.array(e_train)
    x_test = np.array(x_test)
    t_test = np.array(t_test)
    e_test = np.array(e_test)

    from sksurv.metrics import concordance_index_ipcw, brier_score, cumulative_dynamic_auc

    cis = []
    brs = []

    et_train = np.array([(e_train[i], t_train[i]) for i in range(len(e_train))],
                        dtype=[('e', bool), ('t', float)])

    et_test = np.array([(e_test[i], t_test[i]) for i in range(len(e_test))],
                       dtype=[('e', bool), ('t', float)])
    for i, _ in enumerate(tg_test):
        cis.append(concordance_index_ipcw(
            et_train, et_test, out_risk[:, i], tg_test[i])[0])
    brs.append(brier_score(et_train, et_test, out_survival, tg_test)[1])
    roc_auc = []
    for i, _ in enumerate(tg_test):
        roc_auc.append(cumulative_dynamic_auc(
            et_train, et_test, out_risk[:, i], tg_test[i])[0])
    for horizon in enumerate(horizons):
        print(f"For {horizon[1]} quantile")
        print("TD Concordance Index:", cis[horizon[0]])
        print("Brier Score:", brs[0][horizon[0]])
        print("ROC AUC ", roc_auc[horizon[0]][0], "\n")

# %%
out_survival.shape

# %%
x_test.mean(axis=1).shape

# %%
tg_test
