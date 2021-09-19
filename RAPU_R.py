import numpy as np
import scipy as sp
import os
import torch

from data.data_sampler import data_sampler
from data.data_utils import split_train_test, matrics_wrapper
from utils.utils import set_seed, stack_csrdata
from utils.train_utils import get_trainer


def construct_item_pool(normal_matrix, popular_prop):
    popular_filted_item_num = int(normal_matrix.shape[1] * popular_prop)
    item_popularity = np.sum(normal_matrix, axis=0)
    candidate_item = np.argsort(item_popularity, axis=0).reshape(-1)
    candidate_item = candidate_item[-popular_filted_item_num:]
    return candidate_item


def construct_fake_data(fake_data, row, best_proxy_idx):
    if row == -1:
        fake_data[:, best_proxy_idx] = 1
    else:
        row_list = [row] * len(best_proxy_idx)
        fake_data[row_list, best_proxy_idx] = 1
    return fake_data


def local_solution(args,
                   dataset="ml-100k",
                   sample_ratio=0.9,
                   sample_strategy='rw',
                   victim_model="WMFTrainer",
                   fake_user_ratio=0.1,
                   p_item=105,
                   target_items=None,
                   popularity_base=0.1,
                   seed=None,
                   log_print=False):
    if seed is not None:
        args.seed = seed
    set_seed(args.seed)

    print('victim_model name:', victim_model)
    print('fake_user_ratio:', fake_user_ratio)
    print('target items list:', target_items)

    sampled_data = data_sampler(dataset=dataset,
                                sample_ratio=sample_ratio,
                                sample_strategy=sample_strategy,
                                load_save=False)

    split_prop = [0.8, 0.1, 0.1]
    userId, itemId = sampled_data['userId'], sampled_data['itemId']
    rating = sampled_data['rating']
    inter_matrix = sampled_data['inter_matrix']
    inter_matrix[inter_matrix != 0] = 1
    n_users = inter_matrix.shape[0]
    n_items = inter_matrix.shape[1]
    n_fake_users = int(fake_user_ratio * n_users)

    implicit_rating = np.ones_like(rating)
    train_data, val_data, test_data = split_train_test(inter_matrix.shape[0],
                                                       inter_matrix.shape[1],
                                                       userId, itemId,
                                                       implicit_rating,
                                                       split_prop)

    normal_trainer = get_trainer(args,
                                 train_data.shape[0],
                                 inter_matrix.shape[1],
                                 train_data,
                                 model_name=victim_model,
                                 log_print=log_print)
    normal_best_path = normal_trainer.fit(train_data, test_data, val_data)
    recommendations = normal_trainer.recommend(train_data, 100)
    _, prec_df = matrics_wrapper(target_items, recommendations)

    fake_data = np.zeros((n_fake_users, n_items))
    fake_data[:, target_items] = 1.0
    upper = p_item - len(target_items)

    user_embedding, item_embedding = normal_trainer.get_embedding()
    candidate_poll = construct_item_pool(inter_matrix, popularity_base)
    candidata_emb = item_embedding[candidate_poll]
    print(candidate_poll.shape)

    best_proxy_idx_list = []
    for t_item in target_items:
        target_users = np.random.permutation(n_users).tolist()[:int(n_users *
                                                                    0.1)]
        if len(target_users) == 0:
            epsilon = torch.mean(user_embedding, dim=0)
        else:
            epsilon = torch.mean(user_embedding[target_users, :], dim=0)

        target_item_emb = torch.mean(item_embedding[t_item], dim=0)
        p_m = (epsilon + target_item_emb).reshape(1, -1)

        candidate_score = torch.mm(p_m, candidata_emb.t())
        _, best_proxy = torch.topk(candidate_score, upper, dim=1)
        best_proxy_idx = candidate_poll[best_proxy.cpu()]
        best_proxy_idx_list.append(best_proxy_idx)

    for i in range(fake_data.shape[0]):
        fake_data = construct_fake_data(fake_data, i,
                                        best_proxy_idx_list[i % 5])

    set_seed(args.seed, args.use_cuda)
    data = data_sampler(dataset=dataset, sample_ratio=1, load_save=False)

    split_prop = [0.8, 0.1, 0.1]
    n_users = data['num_users']
    n_items = data['num_items']
    userId, itemId, rating = data['userId'], data['itemId'], data['rating']
    inter_matrix = data['inter_matrix']
    inter_matrix[inter_matrix != 0] = 1
    implicit_rating = np.ones_like(rating)
    train_data, val_data, test_data = split_train_test(n_users, n_items,
                                                       userId, itemId,
                                                       implicit_rating,
                                                       split_prop)

    normal_trainer = get_trainer(args,
                                 train_data.shape[0],
                                 n_items,
                                 train_data,
                                 model_name=victim_model,
                                 log_print=log_print)
    normal_best_path = normal_trainer.fit(train_data, test_data, val_data)

    recommendations = normal_trainer.recommend(train_data, 100)
    _, prec_df = matrics_wrapper(target_items, recommendations)
    os.remove("%s.pt" % normal_best_path)

    hyper_train_data = stack_csrdata(train_data,
                                     sp.sparse.csr_matrix(fake_data))
    trainer = get_trainer(args,
                          hyper_train_data.shape[0],
                          n_items,
                          hyper_train_data,
                          model_name=victim_model,
                          log_print=log_print)
    hyper_best_path = trainer.fit(hyper_train_data, test_data, val_data)

    recommendations = trainer.recommend(train_data, 100)
    _, after_df = matrics_wrapper(target_items, recommendations)
    os.remove("%s.pt" % hyper_best_path)

    print('\nBefore attack:')
    print(prec_df)
    print('\nAfter attack:')
    print(after_df)

    return prec_df, after_df
