import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from bunch import Bunch
import warnings
warnings.filterwarnings('ignore', message='')

from trainers.base_trainer import BaseTrainer
from trainers.losses import *
from trainers.losses import pgm_loss, WMW_loss_sigmoid
from utils.utils import sparse2tensor, minibatch


class WeightedMF(nn.Module):
    def __init__(self, n_users, n_items, hidden_dims):
        super(WeightedMF, self).__init__()

        hidden_dims = hidden_dims
        if len(hidden_dims) > 1:
            raise ValueError("WMF can only have one latent dimension.")

        self.n_users = n_users
        self.n_items = n_items
        self.dim = hidden_dims[0]

        self.Q = nn.Parameter(
            torch.zeros([self.n_items, self.dim]).normal_(mean=0, std=0.1))
        self.P = nn.Parameter(
            torch.zeros([self.n_users, self.dim]).normal_(mean=0, std=0.1))
        self.params = nn.ParameterList([self.Q, self.P])

    def forward(self, user_id=None, item_id=None):
        if user_id is None and item_id is None:
            return torch.mm(self.P, self.Q.t())
        if user_id is not None:
            return torch.mm(self.P[[user_id]], self.Q.t())
        if item_id is not None:
            return torch.mm(self.P, self.Q[[item_id]].t())

    def get_embedding(self):
        return self.P.clone().detach(), self.Q.clone().detach()


class WMFTrainer(BaseTrainer):
    def __init__(self,
                 n_users,
                 n_items,
                 args,
                 sigma=1,
                 b=1,
                 barr=1,
                 EM_epoch=10,
                 log_print=True):
        super(WMFTrainer, self).__init__()
        self.args = args

        self.device = torch.device("cuda" if self.args.use_cuda else "cpu")

        self.n_users = n_users
        self.n_items = n_items

        self.metrics = self.args.metrics

        self.log_print = log_print

        self.sigma = sigma
        self.b = b
        self.EM_epoch = EM_epoch
        self.barr = 0.5

        self._initialize()

    def _initialize(self):
        model_args = Bunch(self.args.model)
        if not hasattr(model_args, "optim_method"):
            model_args.optim_method = "sgd"
        hd = model_args.hidden_dims
        self.net = WeightedMF(n_users=self.n_users,
                              n_items=self.n_items,
                              hidden_dims=hd).to(self.device)

        self.optimizer = optim.Adam(self.net.parameters(),
                                    lr=self.args.lr,
                                    weight_decay=self.args.l2)

        self.optim_method = model_args.optim_method
        self.weight_alpha = self.args.model["weight_alpha"]
        self.dim = self.net.dim

    def train_epoch(self, data):
        n_rows = data.shape[0]
        idx_list = np.arange(n_rows)

        model = self.net.to(self.device)
        model.train()
        np.random.shuffle(idx_list)

        epoch_loss = 0.0
        batch_size = (self.args.batch_size
                      if self.args.batch_size > 0 else len(idx_list))
        for batch_idx in minibatch(idx_list, batch_size=batch_size):
            batch_tensor = sparse2tensor(data[batch_idx]).to(self.device)

            outputs = model(user_id=batch_idx)
            loss = mse_loss(data=batch_tensor,
                            logits=outputs,
                            weight=self.weight_alpha).sum()
            epoch_loss += loss.item()

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        return epoch_loss

    def fit_adv(self,
                data_tensor,
                epoch_num,
                unroll_steps,
                n_fakes,
                target_items,
                EM=False):
        if EM:
            return self.fit_adv_EM(data_tensor, epoch_num, unroll_steps,
                                   n_fakes, target_items)
        else:
            return self.fit_adv_sgd(data_tensor, epoch_num, unroll_steps,
                                    n_fakes, target_items)

    def fit_adv_sgd(self, data_tensor, epoch_num, unroll_steps, n_fakes,
                    target_items):
        self._initialize()
        import higher

        if not data_tensor.requires_grad:
            raise ValueError("To compute adversarial gradients, data_tensor "
                             "should have requires_grad=True.")

        data_tensor = data_tensor.to(self.device)
        target_tensor = torch.zeros_like(data_tensor)
        target_tensor[:, target_items] = 1.0
        n_rows = data_tensor.shape[0]
        idx_list = np.arange(n_rows)

        model = self.net.to(self.device)
        optimizer = self.optimizer

        batch_size = (self.args.batch_size
                      if self.args.batch_size > 0 else len(idx_list))
        for i in range(1, epoch_num - unroll_steps + 1):
            t1 = time.time()
            np.random.shuffle(idx_list)
            model.train()
            epoch_loss = 0.0
            for batch_idx in minibatch(idx_list, batch_size=batch_size):
                loss = mse_loss(data=data_tensor[batch_idx],
                                logits=model(user_id=batch_idx),
                                weight=self.weight_alpha).sum()
                epoch_loss += loss.item()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if self.log_print:
                print("Training [{:.1f} s], epoch: {}, loss: {:.4f}".format(
                    time.time() - t1, i, epoch_loss))

        with higher.innerloop_ctx(model, optimizer) as (fmodel, diffopt):
            if self.log_print:
                print("Switching to higher mode...")
            for i in range(epoch_num - unroll_steps + 1, epoch_num + 1):
                t1 = time.time()
                np.random.shuffle(idx_list)
                fmodel.train()
                epoch_loss = 0.0
                for batch_idx in minibatch(idx_list, batch_size=batch_size):
                    loss = mse_loss(data=data_tensor[batch_idx],
                                    logits=fmodel(user_id=batch_idx),
                                    weight=self.weight_alpha).sum()
                    epoch_loss += loss.item()
                    diffopt.step(loss)

            fmodel.eval()
            predictions = fmodel()
            adv_loss = mult_ce_loss(logits=predictions[:-n_fakes, ],
                                    data=target_tensor[:-n_fakes, ]).sum()

            adv_grads = torch.autograd.grad(adv_loss, data_tensor)[0]
            model.load_state_dict(fmodel.state_dict())

        return adv_loss.item(), adv_grads[-n_fakes:, ]

    def fit_adv_EM(self, data_tensor, epoch_num, unroll_steps, n_fakes,
                   target_items):
        self._initialize()
        import higher

        if not data_tensor.requires_grad:
            raise ValueError("To compute adversarial gradients, data_tensor "
                             "should have requires_grad=True.")

        data_tensor = data_tensor.to(self.device)
        target_tensor = torch.zeros_like(data_tensor)
        target_tensor[:, target_items] = 1.0
        n_rows = data_tensor.shape[0]
        idx_list = np.arange(n_rows)

        model = self.net.to(self.device)
        optimizer = self.optimizer

        batch_size = (self.args.batch_size
                      if self.args.batch_size > 0 else len(idx_list))
        for i in range(1, epoch_num - unroll_steps + 1):
            np.random.shuffle(idx_list)
            model.train()
            epoch_loss = 0.0
            for batch_idx in minibatch(idx_list, batch_size=batch_size):
                logits = model(user_id=batch_idx)
                loss = mse_loss(data=data_tensor[batch_idx],
                                logits=logits,
                                weight=self.weight_alpha).sum()
                epoch_loss += loss.item()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        with higher.innerloop_ctx(model, optimizer) as (fmodel, diffopt):
            for i in range(epoch_num - unroll_steps + 1, epoch_num + 1):
                np.random.shuffle(idx_list)
                fmodel.train()
                epoch_loss = 0.0
                for batch_idx in minibatch(idx_list, batch_size=batch_size):
                    fake_user_idx = batch_idx[batch_idx >= (n_rows - n_fakes)]
                    fake_logits = fmodel(user_id=fake_user_idx)
                    fake_user_loss = mse_loss(data=data_tensor[fake_user_idx],
                                              logits=fake_logits,
                                              weight=self.weight_alpha).sum()

                    normal_user_idx = batch_idx[batch_idx < (n_rows - n_fakes)]
                    normal_logits = fmodel(user_id=normal_user_idx)
                    if i <= epoch_num - self.EM_epoch:
                        normal_user_loss = mse_loss(
                            data=data_tensor[normal_user_idx],
                            logits=normal_logits,
                            weight=self.weight_alpha).sum()
                    else:
                        print(".", end='')
                        normal_user_loss = pgm_loss(
                            data=data_tensor[normal_user_idx],
                            logits=normal_logits,
                            weight=self.weight_alpha,
                            sigma=self.sigma,
                            bar_r=self.barr).sum()

                    loss = fake_user_loss + normal_user_loss
                    epoch_loss += loss.item()
                    diffopt.step(loss)

            fmodel.eval()

            loss_type = 'WMW'
            if loss_type == 'WMW':
                predictions = fmodel()
                predictions_ranked, ranking_ind = torch.topk(predictions, 50)
                logits_topK = predictions_ranked[:-n_fakes, ]
                logits_target_items = predictions[:-n_fakes, target_items]
                adv_loss = WMW_loss_sigmoid(
                    logits_topK=logits_topK,
                    logits_target_items=logits_target_items,
                    offset=self.b)
            else:
                predictions = fmodel()
                adv_loss = mult_ce_loss(logits=predictions[:-n_fakes, ],
                                        data=target_tensor[:-n_fakes, ]).sum()

            adv_grads = torch.autograd.grad(adv_loss, data_tensor)[0]
            model.load_state_dict(fmodel.state_dict())

        return adv_loss.item(), adv_grads[-n_fakes:, ]

    def recommend(self, data, top_k, return_preds=False, allow_repeat=False):
        model = self.net.to(self.device)
        model.eval()

        n_rows = data.shape[0]
        idx_list = np.arange(n_rows)
        recommendations = np.empty([n_rows, top_k], dtype=np.int64)
        all_preds = list()
        with torch.no_grad():
            for batch_idx in minibatch(idx_list,
                                       batch_size=self.args.valid_batch_size):
                batch_data = data[batch_idx].toarray()

                preds = model(user_id=batch_idx)
                if return_preds:
                    all_preds.append(preds)
                if not allow_repeat:
                    preds[batch_data.nonzero()] = -np.inf
                if top_k > 0:
                    _, recs = preds.topk(k=top_k, dim=1)
                    recommendations[batch_idx] = recs.cpu().numpy()

        if return_preds:
            return recommendations, torch.cat(all_preds, dim=0).cpu()
        else:
            return recommendations

    def validate(self, train_data, test_data, train_epoch, target_items):
        normal_user_num = test_data.shape[0]

        recommendations = self.recommend(train_data, top_k=100)

        k = 50
        hit_num = 0
        targets = target_items
        topK_rec = recommendations[:normal_user_num, :k]
        for i in range(normal_user_num):
            inter_set_len = len(set(topK_rec[i]).intersection(set(targets)))
            if inter_set_len > 0:
                hit_num += 1
        hit_ratio = hit_num / normal_user_num

        result = {'TargetHR@50': hit_ratio}

        return result

    def get_embedding(self):
        user_embedding_matrix, item_embedding_matrix = self.net.get_embedding()
        return user_embedding_matrix, item_embedding_matrix
