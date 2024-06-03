import sys
sys.path.append('..')
import os
from pycox.evaluation.eval_surv import EvalSurv

from nfm.nfm.eps_config import ParetoEps
from nfm.nfm.base import FullyNeuralNLL
from nfm.nfm.datasets import SurvivalDataset
import torch.nn as nn
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm 



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
        return torch.exp(self.mlp(inputs))
    

import numpy as np


class own_dataset(Dataset):
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




def nfm(dataset, n_iter, learning_rate):
    if dataset == 'mimic':
        trainset = own_dataset(x_path='x_train.npy' , t_path='t_train.npy', e_path='e_train.npy')
        testset = own_dataset(x_path='x_test.npy' , t_path='t_test.npy', e_path='e_test.npy')
        loader = DataLoader(trainset, batch_size=128)
        nll = FullyNeuralNLL(eps_conf=ParetoEps(learnable=True), encoder=Net(num_features = trainset.x.shape[-1])).cuda()
        optimizer = torch.optim.Adam(lr=learning_rate, weight_decay=1e-3, params=nll.parameters())
    elif dataset == 'kkbox':
        trainset = SurvivalDataset.kkbox('train')
        testset = SurvivalDataset.kkbox('test')
        loader = DataLoader(trainset, batch_size=2)
        nll = FullyNeuralNLL(eps_conf=ParetoEps(learnable=True), encoder=Net(num_features = trainset.num_features)).cuda()
        optimizer = torch.optim.Adam(lr=learning_rate, weight_decay=1e-3, params=nll.parameters())

    elif dataset == 'support':
        trainset = own_dataset(x_path='support_x_train.npy' , t_path='support_t_train.npy', e_path='support_e_train.npy')
        testset = own_dataset(x_path='support_x_test.npy' , t_path='support_t_test.npy', e_path='support_e_test.npy')
        loader = DataLoader(trainset, batch_size=128)
        nll = FullyNeuralNLL(eps_conf=ParetoEps(learnable=True), encoder=Net(num_features = trainset.x.shape[-1])).cuda() 
        optimizer = torch.optim.Adam(lr=learning_rate, weight_decay=1e-3, params=nll.parameters())       
    elif dataset == 'synthetic':        
        trainset = own_dataset(x_path='synthetic_x_train.npy' , t_path='synthetic_t_train.npy', e_path='synthetic_e_train.npy')
        testset = own_dataset(x_path='synthetic_x_test.npy' , t_path='synthetic_t_test.npy', e_path='synthetic_e_test.npy')
        loader = DataLoader(trainset, batch_size=128)
        nll = FullyNeuralNLL(eps_conf=ParetoEps(learnable=True), encoder=Net(num_features = trainset.x.shape[-1])).cuda()
        optimizer = torch.optim.Adam(lr=learning_rate, weight_decay=1e-3, params=nll.parameters())   
    else:
        print('dataset error')


    
    for epoch in tqdm(range(n_iter)):
        for i, (x, t, e) in enumerate(loader):
            x = x.to(torch.float32)
            t = t.to(torch.float32)
            e = e.to(torch.float32)
            nll.train()
            if dataset == 'mimic':
                loss = nll(z=x.mean(dim=1).cuda(), y=torch.unsqueeze(t.cuda(),1) / 24., delta=e.cuda())
            elif dataset == 'kkbox':
                loss = nll(z=x.cuda(), y=t.cuda(), delta=e.cuda())
            else:
                loss = nll(z=x.cuda(), y=torch.unsqueeze(t.cuda(),1) , delta=e.cuda())

            # if epoch%100 == 0:
            #     print(loss)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    with torch.no_grad():


        x_test, t_test, e_test = testset[:]
        x_train, t_train, e_train = trainset[:]

        if dataset == 'kkbox':
            x_train = x_train.cpu().numpy()
            t_train = t_train.squeeze().cpu().numpy()
            e_train = e_train.squeeze().cpu().numpy()
            x_test = x_test.cpu().numpy()
            t_test = t_test.cpu().numpy()
            e_test = e_test.cpu().numpy()



        index = np.where(t_test >= t_train.max())
        t_test = np.delete(t_test, index)
        e_test = np.delete(e_test, index)
        x_test = np.delete(x_test, index, axis=0)
        
        horizons = [0.25]

        print(e_train.shape)
        print(e_test.shape)
        
        x = np.concatenate((np.array(x_train), np.array(x_test)), axis=0)
        t = np.concatenate((np.array(t_train), np.array(t_test)), axis=0)
        e = np.concatenate((np.array(e_train), np.array(e_test)), axis=0)

        tg_test = np.quantile(t[e==1], horizons)

        if dataset == 'mimic':
            out_survival = nll.get_survival_prediction( 
                z_test=torch.tensor(x_test.mean(axis=1), dtype=torch.float).cuda(), y_test=torch.tensor(tg_test, dtype=torch.float).view(-1, 1).cuda()).cpu().numpy()
        else:
            out_survival = nll.get_survival_prediction( 
                z_test=torch.tensor(x_test, dtype=torch.float).cuda(), y_test=torch.tensor(tg_test, dtype=torch.float).view(-1, 1).cuda()).cpu().numpy()            
        out_survival = out_survival.T
        out_risk = 1 - out_survival




        from sksurv.metrics import concordance_index_ipcw, brier_score, cumulative_dynamic_auc

        cis = []
        brs = []

        et_train = np.array([(e_train[i], t_train[i]) for i in range(len(e_train))],
                        dtype = [('e', bool), ('t', float)])
        
        et_test = np.array([(e_test[i], t_test[i]) for i in range(len(e_test))],
                        dtype = [('e', bool), ('t', float)])
        for i, _ in enumerate(tg_test):
            cis.append(concordance_index_ipcw(et_train, et_test, out_risk[:, i], tg_test[i])[0])
        brs.append(brier_score(et_train, et_test, out_survival, tg_test)[1])
        roc_auc = []
        for i, _ in enumerate(tg_test):
            roc_auc.append(cumulative_dynamic_auc(et_train, et_test, out_risk[:, i], tg_test[i])[0])
        # for horizon in enumerate(horizons):
        #     print(f"For {horizon[1]} quantile")
        #     print("TD Concordance Index:", cis[horizon[0]])
        #     print("Brier Score:", brs[0][horizon[0]])
        #     print("ROC AUC ", roc_auc[horizon[0]][0], "\n")
        return cis, brs, roc_auc


def result(n_run, dataset, lr):
    cis_list , brs_list, roc_aoc_list = [], [], []
    for j in range(n_run):
        torch.manual_seed(70+j) 
        cis, brs, roc_aoc = nfm(dataset, 20, lr)
        cis_list.append(cis)
        brs_list.append(brs)
        roc_aoc_list.append(roc_aoc)

    report_str = f"""
    Results:
        mean c-index: {np.asarray(cis_list).mean(axis=0)}
        std c-index: {np.asarray(cis_list).std(axis=0)}
        mean bs: {np.asarray(brs_list).mean(axis=0)}
        std bs: {np.asarray(brs_list).std(axis=0)}
        mean roc_aoc: {np.asarray(roc_aoc_list).mean(axis=0)}
        std roc_aoc: {np.asarray(roc_aoc_list).std(axis=0)}
    """
    print(report_str)
    return cis_list, brs_list , roc_aoc_list

cis_list, brs_list , roc_auc_list = result(1, 'kkbox', 1e-6)

np.save('nfm_cis_2.npy', np.array(cis_list))
np.save('nfm_brs_2.npy', np.array(brs_list))
np.save('nfm_roc_auc_2.npy', np.array(roc_auc_list))
