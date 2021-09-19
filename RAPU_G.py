import numpy as np
import time

from utils.utils import set_seed
from data.data_utils import split_train_test

from trainers import BlackBoxAdvTrainer


def run_RAPU_G(data, attack_gen_args, fake_data_path=None):
    set_seed(seed=attack_gen_args.seed)

    num_users = data['num_users']
    num_items = data['num_items']
    num_inter = data['num_inter']
    inter_matrix = data['inter_matrix']
    inter_matrix[inter_matrix != 0] = 1
    user_inter_sum = np.sum(inter_matrix, axis=1)
    user_inter_mean = int(np.mean(user_inter_sum))
    userId, itemId, rating = data['userId'], data['itemId'], data['rating']
    print("num_users: ", num_users, "num_items: ", num_items, "num_inter: ",
          num_inter, "user_inter_mean: ", user_inter_mean)

    split_prop = [0.8, 0.1, 0.1]
    implicit_rating = np.ones_like(rating)
    train_data, val_data, test_data = split_train_test(inter_matrix.shape[0],
                                                       inter_matrix.shape[1],
                                                       userId, itemId,
                                                       implicit_rating,
                                                       split_prop)

    adv_trainer = BlackBoxAdvTrainer(n_users=inter_matrix.shape[0],
                                     n_items=inter_matrix.shape[1],
                                     args=attack_gen_args,
                                     fake_data_path=fake_data_path)
    adv_trainer.fit(train_data, test_data)
