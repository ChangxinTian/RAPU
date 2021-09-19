import os
import time
from collections import OrderedDict

import numpy as np

from utils.utils import save_checkpoint, load_checkpoint


class BaseTrainer(object):
    def __init__(self):
        self.args = None

        self.n_users = None
        self.n_items = None

        self.net = None
        self.optimizer = None
        self.metrics = None
        self.golden_metric = "Recall@50"
        self.log_print = True

    def __repr__(self):
        return self.args["model"]["model_name"] + "_" + str(self.net)

    @property
    def _initialized(self):
        return self.net is not None

    def _initialize(self):
        raise NotImplementedError

    def recommend(self, data, top_k, return_preds=False, allow_repeat=False):
        raise NotImplementedError

    def train_epoch(self, data):
        raise NotImplementedError

    def train_epoch_wrapper(self, train_data, epoch_num):
        time_st = time.time()
        epoch_loss = self.train_epoch(train_data)
        if self.log_print:
            print("Training [{:.1f} s], epoch: {}, loss: {:.4f}".format(
                time.time() - time_st, epoch_num, epoch_loss))

    def evaluate_epoch(self, train_data, test_data, epoch_num):
        t1 = time.time()

        n_rows = train_data.shape[0]
        n_evaluate_users = test_data.shape[0]

        total_metrics_len = sum(len(x) for x in self.metrics)
        total_val_metrics = np.zeros([n_rows, total_metrics_len],
                                     dtype=np.float32)

        recommendations = self.recommend(train_data, top_k=100)

        valid_rows = list()
        for i in range(n_evaluate_users):
            targets = test_data[i].indices
            if targets.size <= 0:
                continue

            recs = recommendations[i].tolist()

            metric_results = list()
            for metric in self.metrics:
                result = metric(targets, recs)
                metric_results.append(result)
            total_val_metrics[i, :] = np.concatenate(metric_results)
            valid_rows.append(i)

        total_val_metrics = total_val_metrics[valid_rows]
        avg_val_metrics = (total_val_metrics.mean(axis=0)).tolist()

        ind, result = 0, OrderedDict()
        for metric in self.metrics:
            values = avg_val_metrics[ind:ind + len(metric)]
            if len(values) <= 1:
                result[str(metric)] = values[0]
            else:
                for name, value in zip(str(metric).split(','), values):
                    result[name] = value
            ind += len(metric)

        print_key_10 = "Recall@10"
        print_key_50 = "Recall@50"
        if self.log_print:
            print("Evaluate [{:.1f} s],  epoch: {}, {}:{} , {}:{}".format(
                time.time() - t1, epoch_num, print_key_10,
                str(result[print_key_10]), print_key_50,
                str(result[print_key_50])))
        return result

    def fit(self, train_data, test_data, val_data):
        if not self._initialized:
            self._initialize()

        if self.args.save_feq > self.args.epochs:
            raise ValueError("Model save frequency should be smaller than"
                             " total training epochs.")

        start_epoch = 0
        best_checkpoint_path = ""
        best_perf = 0.0
        time_point = time.strftime("-%Y-%m-%d-%H:%M:%S", time.localtime())
        checkpoint_path = os.path.join(
            self.args.output_dir,
            self.args.model['model_name'] + time_point + str(time.time()))
        print("Saving checkpoint to ", checkpoint_path)
        for epoch_num in range(start_epoch, self.args.epochs):
            self.train_epoch_wrapper(train_data, epoch_num)
            if epoch_num % self.args.save_feq == 0:
                print(epoch_num, end=" ")
                result = self.evaluate_epoch(train_data, val_data, epoch_num)
                if result[self.golden_metric] > best_perf:
                    str_metric = "{}={:.4f}".format(self.golden_metric,
                                                    result[self.golden_metric])
                    if self.log_print:
                        print("Having better model checkpoint with"
                              " performance {}".format(str_metric))

                    save_checkpoint(self.net,
                                    self.optimizer,
                                    checkpoint_path,
                                    epoch=epoch_num)

                    best_perf = result[self.golden_metric]
                    best_checkpoint_path = checkpoint_path

        print("Loading best model checkpoint.")
        self.restore(best_checkpoint_path)
        result = self.evaluate_epoch(train_data, test_data, -1)
        print('Recall@10: ', result['Recall@10'])
        return best_checkpoint_path

    def restore(self, path):
        start_epoch, model_state, optimizer_state = load_checkpoint(path)
        self.net.load_state_dict(model_state)
        self.optimizer.load_state_dict(optimizer_state)
        return start_epoch
