import os
import time

import numpy as np
import torch
import torch.optim as optim
from bunch import Bunch
from scipy import sparse

from utils.utils import set_seed, sparse2tensor, tensor2sparse, stack_csrdata, save_fake_data


class BlackBoxAdvTrainer:
    def __init__(self, n_users, n_items, args, fake_data_path=None):
        self.args = args
        self.device = torch.device("cuda" if self.args.use_cuda else "cpu")

        self.n_users = n_users
        self.n_items = n_items
        self.n_fakes = int(n_users * args.fake_user_ratio)
        self.target_items = args.target_items
        self.golden_metric = "TargetHR@50"

        self.EM = args.EM
        self.sigma = args.sigma
        self.b = args.b
        self.proj_topk = args.proj_topk
        self.EM_epoch = args.EM_epoch
        self.p_item = args.item
        self.mix_project = args.popularity_smooth
        self.fake_data_path = fake_data_path

        print(self.args.proj_threshold)

    def __repr__(self):
        return ("Random" if self.args.attack_type == "random" else
                self.args.surrogate["model"]["model_name"])

    def _initialize(self, train_data):
        fake_data = self.init_fake_data(train_data=train_data)

        self.fake_tensor = sparse2tensor(fake_data)
        self.fake_tensor.requires_grad_()

        self.optimizer = optim.SGD([self.fake_tensor],
                                   lr=self.args.adv_lr,
                                   momentum=self.args.adv_momentum)

    def train_epoch(self, train_data, epoch_num):
        def compute_adv_grads():
            sur_args = Bunch(self.args.surrogate)
            sur_args.model_load_path = ""
            sur_trainer_class = sur_args.model["trainer_class"]
            sur_trainer_ = sur_trainer_class(n_users=self.n_users +
                                             self.n_fakes,
                                             n_items=self.n_items,
                                             args=sur_args,
                                             sigma=self.sigma,
                                             b=self.b,
                                             EM_epoch=self.EM_epoch,
                                             log_print=False)

            data_tensor = torch.cat([
                sparse2tensor(train_data).to(self.device),
                self.fake_tensor.detach().clone().to(self.device)
            ],
                                    dim=0)
            data_tensor.requires_grad_()

            adv_loss_, adv_grads_ = sur_trainer_.fit_adv(
                data_tensor=data_tensor,
                epoch_num=sur_args["epochs"],
                unroll_steps=self.args.unroll_steps,
                n_fakes=self.n_fakes,
                target_items=self.target_items,
                EM=self.EM)
            return sur_trainer_, adv_loss_, adv_grads_

        def project_tensor(fake_tensor, threshold=0.5):
            if not self.proj_topk:
                return torch.where(fake_tensor > threshold,
                                   torch.ones_like(fake_tensor),
                                   torch.zeros_like(fake_tensor))
            else:
                upper = self.p_item
                values, idxs = torch.topk(fake_tensor, upper, dim=1)
                user_idxs = torch.tensor(range(fake_tensor.shape[0])).reshape(
                    -1, 1).repeat(1, upper).reshape(-1)
                item_idxs = idxs.reshape(-1)
                new_fake_tensor = torch.zeros_like(fake_tensor)
                new_fake_tensor[user_idxs, item_idxs] = 1
                return new_fake_tensor

        def popularity_enhanced_project_tensor(fake_tensor, threshold=0.5):
            beta = 1
            popularity = torch.div(self.train_data_popularity,
                                   torch.max(self.train_data_popularity))
            fake_tensor = fake_tensor + beta * popularity
            upper = self.p_item
            values, idxs = torch.topk(fake_tensor, upper, dim=1)
            user_idxs = torch.tensor(range(fake_tensor.shape[0])).reshape(
                -1, 1).repeat(1, upper).reshape(-1)
            item_idxs = idxs.reshape(-1)
            new_fake_tensor = torch.zeros_like(fake_tensor)
            new_fake_tensor[user_idxs, item_idxs] = 1
            return new_fake_tensor

        sur_trainer = None
        new_fake_tensor = None
        if self.args.attack_type == "random":
            new_fake_tensor = project_tensor(
                fake_tensor=torch.randn_like(self.fake_tensor),
                threshold=self.args.proj_threshold)

            if self.args.click_targets:
                new_fake_tensor[:, self.target_items] = 1.0
        elif self.args.attack_type == "adversarial":
            t1 = time.time()
            set_seed(self.args.seed, cuda=self.args.use_cuda, log=False)
            self.optimizer.zero_grad()

            sur_trainer, adv_loss, adv_grads = compute_adv_grads()
            if self.args.click_targets:
                adv_grads[:, self.target_items] = 0.0
            print(
                "\nAdversarial training [{:.1f} s],  epoch: {}, loss: {:.4f}".
                format(time.time() - t1, epoch_num, adv_loss),
                end='\t\t')

            normalized_adv_grads = adv_grads / adv_grads.norm(
                p=2, dim=1, keepdim=True)
            if self.fake_tensor.grad is None:
                self.fake_tensor.grad = normalized_adv_grads.detach().cpu()
            else:
                self.fake_tensor.grad.data = normalized_adv_grads.detach().cpu(
                )

            self.optimizer.step()

            if self.mix_project:
                new_fake_tensor = popularity_enhanced_project_tensor(
                    fake_tensor=self.fake_tensor.data.clone(),
                    threshold=self.args.proj_threshold)
            else:
                new_fake_tensor = project_tensor(
                    fake_tensor=self.fake_tensor.data.clone(),
                    threshold=self.args.proj_threshold)

            if self.args.click_targets:
                new_fake_tensor[:, self.target_items] = 1.0

        return sur_trainer, new_fake_tensor

    def evaluate_epoch(self, trainer, train_data, test_data):
        result = trainer.validate(train_data=train_data,
                                  test_data=test_data,
                                  train_epoch=-1,
                                  target_items=self.target_items)
        return result

    def fit(self, train_data, test_data):
        self._initialize(train_data)

        if self.args.attack_type not in ["adversarial", "random"]:
            raise ValueError("Unknown attack type {}.".format(
                self.args.attack_type))

        best_fake_data, best_perf = None, 0.0
        if self.args.attack_type == "random":
            cur_fake_tensor = self.fake_tensor.detach().clone()
            cur_sur_trainer, random_fake_tensor = self.train_epoch(
                train_data, -1)
            print("Total changes: {}".format(
                (random_fake_tensor - cur_fake_tensor).abs().sum().item()),
                  end='\t\t')
            best_fake_data = tensor2sparse(random_fake_tensor)
        elif self.args.attack_type == "adversarial":
            sparse_popularity = train_data.sum(0)
            self.train_data_popularity = torch.from_numpy(sparse_popularity)
            cur_fake_tensor = self.fake_tensor.detach().clone()
            for epoch_num in range(1, self.args.adv_epochs + 1):
                cur_sur_trainer, new_fake_tensor = self.train_epoch(
                    train_data, epoch_num)
                print("Total changes: {}".format(
                    (new_fake_tensor - cur_fake_tensor).abs().sum().item()),
                      end='\t\t')

                self.fake_tensor.data = new_fake_tensor.detach().clone()
                cur_fake_tensor = new_fake_tensor.detach().clone()

                cur_fake_data = tensor2sparse(cur_fake_tensor)
                result = self.evaluate_epoch(trainer=cur_sur_trainer,
                                             train_data=stack_csrdata(
                                                 train_data, cur_fake_data),
                                             test_data=test_data)

                cur_perf = result[self.golden_metric]
                print("Hit@50:%.2f" % cur_perf, end='\t\t')
                if cur_perf > best_perf:
                    print("Better fake data with "
                          "{}={:.4f}".format(self.golden_metric, cur_perf),
                          end='\t\t')
                    best_fake_data, best_perf = cur_fake_data, cur_perf

        if self.fake_data_path:
            fake_data_path = self.fake_data_path
        else:
            fake_data_path = os.path.join(
                self.args.output_dir,
                "_".join([str(self), "fake_data", "best"]))

        print('\nSave to ', fake_data_path)
        print('HR@50 on Sur Model: ', best_perf)
        print('rating num of each fake user: ',
              best_fake_data.nnz / self.n_fakes)
        save_fake_data(best_fake_data, path=fake_data_path)

    def init_fake_data(self, train_data):
        train_data = train_data.toarray()
        max_allowed_click = 100
        user_clicks = train_data.sum(1)
        qual_users = np.where(user_clicks <= max_allowed_click)[0]

        indices = np.arange(len(qual_users))
        np.random.shuffle(indices)
        sampled_users = qual_users[:self.n_fakes]
        fake_data = sparse.csr_matrix(train_data[sampled_users],
                                      dtype=np.float64,
                                      shape=(self.n_fakes, self.n_items))
        return fake_data
