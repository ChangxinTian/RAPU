import numpy as np
import pandas as pd
import torch
from scipy import sparse

EPSILON = 1e-12


def get_target_item_list(dataset='ml-100k'):
    ml_target_item_list = [1018, 946, 597, 575, 516]
    video_target_item_list = [3183, 3648, 1703, 2821, 1051]

    if dataset == 'ml-100k':
        return ml_target_item_list
    if dataset == 'Video':
        return video_target_item_list
    print('This dataset is not available')
    return 0


def split_train_test(n_users, n_items, userId, itemId, rating, split_prop):
    inter = np.concatenate((np.expand_dims(userId, 1), np.expand_dims(
        itemId, 1), np.expand_dims(rating, 1)),
                           axis=1)
    np.random.shuffle(inter)
    inter_num = inter.shape[0]

    train_inter = inter[:int(inter_num * split_prop[0]), :]
    val_inter = inter[int(inter_num * split_prop[0]):int(inter_num *
                                                         (split_prop[0] +
                                                          split_prop[1])), :]
    test_inter = inter[int(inter_num * (split_prop[0] + split_prop[1])):, :]

    train_data = sparse.csr_matrix((np.ones_like(train_inter[:, 2]),
                                    (train_inter[:, 0], train_inter[:, 1])),
                                   dtype='float64',
                                   shape=(n_users, n_items))
    val_data = sparse.csr_matrix(
        (np.ones_like(val_inter[:, 2]), (val_inter[:, 0], val_inter[:, 1])),
        dtype='float64',
        shape=(n_users, n_items))
    test_data = sparse.csr_matrix(
        (np.ones_like(test_inter[:, 2]), (test_inter[:, 0], test_inter[:, 1])),
        dtype='float64',
        shape=(n_users, n_items))

    return train_data, val_data, test_data


def matrics_wrapper(targets, predictions, k=[10, 20, 50, 100]):
    target = targets
    recommendations = predictions
    matric = Matrics(k=[10, 20, 50, 100])

    n_evaluate_users = predictions.shape[0]
    n_matric = matric.matric
    matric_result = np.zeros((n_evaluate_users, n_matric * len(k)))

    for i in range(n_evaluate_users):
        recs = recommendations[i].tolist()

        _, result_arr = matric(target, recs)  # shape(4,6)
        matric_result[i, :] = result_arr.reshape(1, -1)

    avg_val_metrics = (matric_result.mean(axis=0))
    avg_val_metrics = avg_val_metrics.reshape(-1, n_matric)

    avg_val_metrics_df = pd.DataFrame(
        avg_val_metrics,
        columns=['Precision', 'Recall', 'F1', 'Hit', 'MeanAP', 'NDCG', 'MRR'],
        index=k)

    return avg_val_metrics, avg_val_metrics_df


class Matrics:
    def __init__(self, k=[10, 20, 50, 100]):
        self.k = k
        self.matric = 7

    def __len__(self):
        return len(self.k)

    def __call__(self, targets, predictions):
        result_dic = {}
        result_arr = np.zeros((len(self.k), self.matric))
        for r, k in enumerate(self.k):
            result = {}
            if len(predictions) < k:
                print("length of prediction less than k")
            k_predictions = predictions[:k]
            num_hit = len(set(k_predictions).intersection(set(targets)))
            precision = float(num_hit) / len(k_predictions)
            recall = float(num_hit) / len(targets)
            hit = 1 if num_hit > 0 else 0

            if precision and recall:
                F1 = (2 * precision * recall) / (precision + recall)
            else:
                F1 = 0

            score = 0.0
            num_hits = 0.0
            for i, p in enumerate(k_predictions):
                if p in targets:
                    num_hits += 1.0
                    score += num_hits / (i + 1.0)
            MeanAP = score / k

            idcg = np.sum(1 / np.log2(np.arange(2, len(targets) + 2)))
            dcg = 0.0
            for i, p in enumerate(k_predictions):
                if p in targets:
                    dcg += 1 / np.log2(i + 2)
            ndcg = dcg / idcg

            MRR = 0
            for idx, item in enumerate(k_predictions):
                if item in targets:
                    MRR = 1 / (idx + 1)
                    break

            result = {
                'precision@{}'.format(k): precision,
                'recall@{}'.format(k): recall,
                'F1@{}'.format(k): F1,
                'hit@{}'.format(k): hit,
                'MeanAP@{}'.format(k): MeanAP,
                'NDCG@{}'.format(k): ndcg,
                'MRR@{}'.format(k): MRR
            }
            result_dic[k] = result
            result_arr[r, :] = np.array(
                [precision, recall, F1, hit, MeanAP, ndcg, MRR])

        return result_dic, result_arr
