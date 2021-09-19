# -*- coding: utf-8 -*-
import numpy as np
import os

from data.data_sampler import data_sampler
from data.data_utils import split_train_test, matrics_wrapper, get_target_item_list
from utils.utils import set_seed, stack_csrdata, load_fake_data
from utils.train_utils import get_trainer
from args.args_utils import get_args


def evalute_attack(fake_data_path,
                   model_name="WMFTrainer",
                   dataset="ml-100k",
                   seed=None,
                   log_print=False):
    args = get_args(dataset)
    if seed is not None:
        args.seed = seed

    set_seed(args.seed, args.use_cuda)
    targets = get_target_item_list(dataset=dataset)

    data = data_sampler(dataset=dataset, sample_ratio=1, load_save=False)
    n_users = data['num_users']
    n_items = data['num_items']

    split_prop = [0.8, 0.1, 0.1]
    userId, itemId, rating = data['userId'], data['itemId'], data['rating']
    implicit_rating = np.ones_like(rating)
    train_data, val_data, test_data = split_train_test(n_users, n_items,
                                                       userId, itemId,
                                                       implicit_rating,
                                                       split_prop)

    normal_trainer = get_trainer(args,
                                 train_data.shape[0],
                                 n_items,
                                 train_data,
                                 model_name=model_name,
                                 log_print=log_print)
    normal_best_path = normal_trainer.fit(train_data, test_data, val_data)

    recommendations = normal_trainer.recommend(train_data, 100)
    _, prec_df = matrics_wrapper(targets, recommendations)
    os.remove("%s.pt" % normal_best_path)

    fake_data = load_fake_data(fake_data_path)
    fake_data[:, targets] = 1
    hyper_train_data = stack_csrdata(train_data, fake_data)

    trainer = get_trainer(args,
                          hyper_train_data.shape[0],
                          n_items,
                          hyper_train_data,
                          model_name=model_name,
                          log_print=log_print)
    hyper_best_path = trainer.fit(hyper_train_data, test_data, val_data)

    recommendations = trainer.recommend(train_data, 100)
    _, after_df = matrics_wrapper(targets, recommendations)
    os.remove("%s.pt" % hyper_best_path)

    print('\nBefore attack:')
    print(prec_df)
    print('\nAfter attack:')
    print(after_df)

    return prec_df, after_df
