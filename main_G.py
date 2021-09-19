import argparse
import time
import os

from data.data_sampler import data_sampler
from data.data_utils import get_target_item_list
from args.args_utils import get_attack_args
from utils.utils import set_seed
from evalute_attack import evalute_attack
from RAPU_G import run_RAPU_G

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

parser = argparse.ArgumentParser()

parser.add_argument('--dataset', '-d', type=str, default="ml-100k")
parser.add_argument('--sample_ratio', '-sr', type=float, default=0.9)
parser.add_argument('--sample_strategy', '-ss', type=str, default='rw')
parser.add_argument('--load_save', type=int, default=0)

parser.add_argument('--victim_model', '-v', type=str, default='WMFTrainer')
parser.add_argument('--fake_user_ratio', '-f', type=float, default=0.005)
parser.add_argument('--item', '-i', type=int, default=100)

parser.add_argument('--proj_threshold', '-pt', type=float, default=0)
parser.add_argument('--proj_topk', '-pk', type=int, default=1)
parser.add_argument('--popularity_smooth', '-ps', type=int, default=1)
parser.add_argument('--WMW_b', '-b', type=float, default=0.1)

parser.add_argument('--EM', '-em', type=int, default=1)
parser.add_argument('--EM_epoch', '-ee', type=int, default=3)
parser.add_argument('--sigma', type=float, default=0.3)
parser.add_argument('--seed', type=int, default=1234)

args = parser.parse_args()
print("Argument: ")
print(args)

attack_gen_args = get_attack_args(args.dataset)
attack_gen_args.seed = args.seed
attack_gen_args.fake_user_ratio = args.fake_user_ratio
attack_gen_args.proj_threshold = args.proj_threshold
attack_gen_args.proj_topk = True if args.proj_topk == 1 else False
attack_gen_args.popularity_smooth = True if args.popularity_smooth == 1 else False
attack_gen_args.b = args.WMW_b
attack_gen_args.EM = True if args.EM == 1 and args.EM_epoch > 0 else False
attack_gen_args.EM_epoch = args.EM_epoch
attack_gen_args.sigma = args.sigma
attack_gen_args.target_items = get_target_item_list(dataset=args.dataset)
attack_gen_args.item = args.item + len(attack_gen_args.target_items)
load_save = True if args.load_save == 1 else False

s = 0
set_seed(seed=s)
attack_gen_args.seed = s

t = time.time()
fake_data_path = "./fakedata/our_{}_sampleStrategy{}_sampleRatio{}_userRatio{}_itemNum{}_EMepoch{}_seed{}.npz".format(
    args.dataset, str(args.sample_strategy), str(args.sample_ratio),
    str(args.fake_user_ratio), str(args.item), str(args.EM_epoch), str(s))

data = data_sampler(dataset=args.dataset,
                    sample_ratio=args.sample_ratio,
                    sample_strategy=args.sample_strategy,
                    load_save=False,
                    seed=s)
run_RAPU_G(data, attack_gen_args, fake_data_path)

prec_df, after_df = evalute_attack(fake_data_path=fake_data_path,
                                   model_name=args.victim_model,
                                   dataset=args.dataset,
                                   seed=s)
if not load_save:
    os.remove(fake_data_path)

print('time cost: ', int(time.time() - t) / 60)
