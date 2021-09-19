import numpy as np
import pandas as pd
import os
import random


def data_process(dataset='ml-100k', k=10, load_save=False):
    if dataset == 'ml-100k':
        data = preprocessing_ml100k(file_loc='./data/full/ml-100k/u.data')
    elif dataset == 'Video':
        file_loc = './data/full/instant-video/ratings_Amazon_Instant_Video.csv'
        data = preprocessing_amazon(file_loc=file_loc,
                                    k=k,
                                    load_save=load_save)
    else:
        print("have no this dataset, the default is ml-100k. ")
        data = preprocessing_ml100k(file_loc='./data/full/ml-100k/u.data')
    return data


def preprocessing_ml100k(file_loc='./data/full/ml-100k/u.data'):
    data = []
    with open(file_loc, 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.split('\t')
            data_line = [int(line[0]), int(line[1]), int(line[2])]
            data.append(data_line)

    data = np.asarray(data)
    userId = data[:, 0] - 1
    itemId = data[:, 1] - 1
    rating = data[:, 2]

    num_users = len(set(userId))
    num_items = len(set(itemId))
    num_inter = rating.shape[0]
    print('num_users:{}\t\tnum_items:{}\t\tnum_inter:{}'.format(
        num_users, num_items, num_inter))
    print('Sparsity: {}'.format(1 - num_inter / (num_users * num_items)))

    rating_max = rating.max()
    rating_min = rating.min()

    inter_matrix = np.zeros((num_users, num_items))
    inter_matrix[userId, itemId] = rating

    item_rating = [[] for _ in range(num_items)]
    for j, y in zip(itemId, rating):
        item_rating[j].append(y)
    item_mean_rating = [np.mean(i_rating) for i_rating in item_rating]
    item_var_rating = [np.var(i_rating) for i_rating in item_rating]

    return userId, itemId, rating, inter_matrix, item_mean_rating, item_var_rating, \
        num_users, num_items, num_inter, rating_max, rating_min


def preprocessing_amazon(
        file_loc='./full/instant-video/ratings_Amazon_Instant_Video.csv',
        k=10,
        load_save=False):
    path_tmp = "_core{}.npz".format(str(k))
    save_path = file_loc + path_tmp

    print("\nfile_loc:", file_loc)
    print("save_path: ", save_path)
    print("k-core: ", k)

    if load_save and os.path.exists(save_path):
        X = np.load(save_path)
        X_arr = [X[i] for i in X]
        userId, itemId, rating, inter_matrix, item_mean_rating, item_var_rating, num_users, num_items, num_inter, rating_max, rating_min = X_arr
        num_users, num_items = num_users.item(), num_items.item()
        num_inter = num_inter.item()
        rating_max, rating_min = rating_max.item(), rating_min.item()
    else:
        file = pd.read_csv(file_loc,
                           header=None,
                           names=['user', 'item', 'rating', 'timestamp'])
        userId_str = file['user'].values
        itemId_str = file['item'].values
        raw_rating = file['rating'].values

        raw_num_users = len(set(userId_str))
        raw_num_items = len(set(itemId_str))
        raw_num_inter = raw_rating.shape[0]
        print(
            'raw_num_users:{}\traw_num_items:{}\traw_num_inter:{}\tsparsity:{}'
            .format(raw_num_users, raw_num_items, raw_num_inter,
                    1 - raw_num_inter / (raw_num_users * raw_num_items)))

        user_inter_count = {}
        item_inter_count = {}
        for u, i in zip(userId_str, itemId_str):
            user_inter_count[u] = user_inter_count.setdefault(u, 0) + 1
            item_inter_count[i] = item_inter_count.setdefault(i, 0) + 1
        user_str2id_dict = {}
        item_str2id_dict = {}
        userId = []
        itemId = []
        rating = []
        for u, i, r in zip(userId_str, itemId_str, raw_rating):
            if user_inter_count[u] < k or item_inter_count[i] < k:
                continue
            if u not in user_str2id_dict:
                user_str2id_dict[u] = len(user_str2id_dict)
            if i not in item_str2id_dict:
                item_str2id_dict[i] = len(item_str2id_dict)
            userId.append(user_str2id_dict[u])
            itemId.append(item_str2id_dict[i])
            rating.append(r)
        userId = np.array(userId, dtype=np.int32)
        itemId = np.array(itemId, dtype=np.int32)
        rating = np.array(rating)

        num_users = len(set(userId))
        num_items = len(set(itemId))
        num_inter = rating.shape[0]

        rating_max = rating.max()
        rating_min = rating.min()

        inter_matrix = np.zeros((num_users, num_items))
        inter_matrix[userId, itemId] = rating

        item_rating = [[] for _ in range(num_items)]
        for j, y in zip(itemId, rating):
            item_rating[j].append(y)

        item_mean_rating = np.array(
            [np.mean(i_rating) for i_rating in item_rating])
        item_var_rating = np.array(
            [np.var(i_rating) for i_rating in item_rating])

        np.savez(save_path, userId, itemId, rating, inter_matrix,
                 item_mean_rating, item_var_rating, num_users, num_items,
                 num_inter, rating_max, rating_min)

    print('num_users:{}\t\tnum_items:{}\t\tnum_inter:{}\t\tsparsity:{}'.format(
        num_users, num_items, num_inter,
        1 - num_inter / (num_users * num_items)))
    return userId, itemId, rating, inter_matrix, item_mean_rating, item_var_rating, \
        num_users, num_items, num_inter, rating_max, rating_min


def data_sampler(dataset='ml-100k',
                 sample_ratio=0.9,
                 sample_strategy='rw',
                 load_save=False,
                 k=10,
                 jump_prop=0.1,
                 seed=1234):
    if sample_ratio == 1.0:
        data = data_process(dataset, k, load_save)
        userId, itemId, rating, inter_matrix, item_mean_rating, item_var_rating, num_users, num_items, num_inter, rating_max, rating_min = data
        data_dict = {
            'userId': userId,
            'itemId': itemId,
            'rating': rating,
            'inter_matrix': inter_matrix,
            'item_mean_rating': item_mean_rating,
            'item_var_rating': item_var_rating,
            'num_users': num_users,
            'num_items': num_items,
            'num_inter': num_inter,
            'rating_max': rating_max,
            'rating_min': rating_min
        }
        return data_dict

    print("k:{}\tsample_ratio:{}\tsample_strategy:{}\tseed:{}".format(
        k, sample_ratio, sample_strategy, seed))

    save_path = "./data/partial/" + dataset + "_{}_k{}_ratio{}_jump{}_seed{}.npz".format(
        sample_strategy, k, sample_ratio, jump_prop, seed)
    print("save_path:{}".format(save_path))

    sampled_userId, sampled_itemId, sampled_rating = None, None, None
    sampled_inter_matrix = None
    sampled_item_mean_rating, sampled_item_var_rating = None, None
    sampled_num_users, sampled_num_items, sampled_num_inter = None, None, None
    sampled_rating_max, sampled_rating_min = None, None
    if load_save and os.path.exists(save_path):
        X = np.load(save_path)
        X_arr = [X[i] for i in X]
        sampled_userId, sampled_itemId, sampled_rating, sampled_inter_matrix, sampled_item_mean_rating, sampled_item_var_rating, sampled_num_users, sampled_num_items, sampled_num_inter, sampled_rating_max, sampled_rating_min = X_arr
        sampled_num_users = sampled_num_users.item()
        sampled_num_items = sampled_num_items.item()
        sampled_num_inter = sampled_num_inter.item()
        sampled_rating_max = sampled_rating_max.item()
        sampled_rating_min = sampled_rating_min.item()
    else:
        data = data_process(dataset, k, load_save)
        userId, itemId, rating, inter_matrix, _, _, num_users, num_items, num_inter, _, _ = data
        sampled_userId, sampled_itemId, sampled_rating = [], [], []
        sampled_inter_matrix = np.zeros((num_users, num_items))

        if sample_strategy == 'rw':  # random walk with jump: replace part of ture positive with false positive
            print("Sample by random walk with jump ...")
            start_point = random.sample(range(num_users), int(num_users * 0.1))
            num_start_point = len(list(start_point))
            num_ture_postive = sample_ratio * num_inter
            num_false_postive = (1 - sample_ratio) * num_inter
            cur_num_ture_postive, cur_num_false_postive = 0, 0
            all_item_set = set(range(num_items))
            all_user_set = set(range(num_users))
            is_true_pos = True
            while cur_num_ture_postive < num_ture_postive or cur_num_false_postive < num_false_postive:
                for _ in range(num_start_point):
                    s = start_point.pop(0)
                    s_inter = inter_matrix[s, :]
                    s_inter_idx = np.argwhere(s_inter > 0)[:, 0].tolist()
                    if np.random.rand() < jump_prop:
                        is_true_pos = False
                        next_i = random.sample(
                            list(all_item_set - set(s_inter_idx)), 1)[0]
                    else:
                        is_true_pos = True
                        next_i = random.sample(s_inter_idx, 1)[0]

                    if (is_true_pos
                            and cur_num_ture_postive < num_ture_postive) or (
                                (not is_true_pos)
                                and cur_num_false_postive < num_false_postive):
                        if sampled_inter_matrix[s, next_i] == 0:
                            sampled_userId.append(s)
                            sampled_itemId.append(next_i)
                            sampled_rating.append(inter_matrix[s, next_i])
                            sampled_inter_matrix[s, next_i] = 1
                            if is_true_pos:
                                cur_num_ture_postive += 1
                            else:
                                cur_num_false_postive += 1
                    start_point.append(next_i)

                for _ in range(num_start_point):
                    i = start_point.pop(0)
                    i_inter = inter_matrix[:, i]
                    i_inter_idx = np.argwhere(i_inter > 0)[:, 0].tolist()
                    if np.random.rand() < jump_prop:  # add false positive
                        is_true_pos = False
                        next_u = random.sample(
                            list(all_user_set - set(i_inter_idx)), 1)[0]
                    else:
                        is_true_pos = True
                        next_u = random.sample(i_inter_idx, 1)[0]

                    if (is_true_pos
                            and cur_num_ture_postive < num_ture_postive) or (
                                (not is_true_pos)
                                and cur_num_false_postive < num_false_postive):
                        if sampled_inter_matrix[next_u, i] == 0:
                            sampled_userId.append(next_u)
                            sampled_itemId.append(i)
                            sampled_rating.append(inter_matrix[next_u, i])
                            sampled_inter_matrix[next_u, i] = 1
                            if is_true_pos:
                                cur_num_ture_postive += 1
                            else:
                                cur_num_false_postive += 1
                    start_point.append(next_u)
        elif sample_strategy == 'random':
            print("Sample by random  ... ", sample_ratio)
            _ = random.sample(range(num_users), int(num_users * 0.1))
            false_pos = 0
            for u, i, r in zip(userId, itemId, rating):
                if np.random.rand() > sample_ratio:
                    while True:
                        u = random.randint(0, num_users - 1)
                        i = random.randint(0, num_items - 1)
                        if inter_matrix[u, i] < 0.5:
                            false_pos += 1
                            break
                sampled_userId.append(u)
                sampled_itemId.append(i)
                sampled_rating.append(r)
                sampled_inter_matrix[u, i] = 1
            print('false_pos:', false_pos)

    sampled_num_users = len(set(sampled_userId))
    sampled_num_items = len(set(sampled_itemId))
    sampled_num_inter = len(sampled_rating)

    sampled_userId = np.array(sampled_userId, dtype=np.int32)
    sampled_itemId = np.array(sampled_itemId, dtype=np.int32)
    sampled_rating = np.array(sampled_rating)

    sampled_rating_max = sampled_rating.max()
    sampled_rating_min = sampled_rating.min()

    sampled_item_rating = [[] for _ in range(sampled_inter_matrix.shape[1])]
    for j, y in zip(sampled_itemId, sampled_rating):
        sampled_item_rating[j].append(y)

    sampled_item_mean_rating = np.array([
        np.mean(i_rating) if len(i_rating) > 0 else 0
        for i_rating in sampled_item_rating
    ])
    sampled_item_var_rating = np.array([
        np.var(i_rating) if len(i_rating) > 0 else 0
        for i_rating in sampled_item_rating
    ])

    np.savez(save_path, sampled_userId, sampled_itemId, sampled_rating,
             sampled_inter_matrix, sampled_item_mean_rating,
             sampled_item_var_rating, sampled_num_users, sampled_num_items,
             sampled_num_inter, sampled_rating_max, sampled_rating_min)
    data_dict = {
        'userId': sampled_userId,
        'itemId': sampled_itemId,
        'rating': sampled_rating,
        'inter_matrix': sampled_inter_matrix,
        'item_mean_rating': sampled_item_mean_rating,
        'item_var_rating': sampled_item_var_rating,
        'num_users': sampled_num_users,
        'num_items': sampled_num_items,
        'num_inter': sampled_num_inter,
        'rating_max': sampled_rating_max,
        'rating_min': sampled_rating_min
    }
    return data_dict


def load_npz(path):
    X = np.load(path)
    X_arr = [X[i] for i in X]
    userId, itemId, rating, inter_matrix, item_mean_rating, item_var_rating, num_users, num_items, num_inter, rating_max, rating_min = X_arr
    num_users, num_items = num_users.item(), num_items.item()
    num_inter = num_inter.item()
    rating_max, rating_min = rating_max.item(), rating_min.item()
    print('num_users:{}\t\tnum_items:{}\t\tnum_inter:{}'.format(
        num_users, num_items, num_inter))
    print('Sparsity: {}'.format(1 - num_inter / (num_users * num_items)))
    data_dict = {
        'userId': userId,
        'itemId': itemId,
        'rating': rating,
        'inter_matrix': inter_matrix,
        'item_mean_rating': item_mean_rating,
        'item_var_rating': item_var_rating,
        'num_users': num_users,
        'num_items': num_items,
        'num_inter': num_inter,
        'rating_max': rating_max,
        'rating_min': rating_min
    }
    return data_dict
