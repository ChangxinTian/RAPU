import argparse
import time
import os

from data.data_utils import get_target_item_list
from args.args_utils import get_args
from RAPU_R import local_solution

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

parser = argparse.ArgumentParser()

parser.add_argument('--dataset', '-d', type=str, default="ml-100k")
parser.add_argument('--sample_ratio', '-sr', type=float, default=0.90)
parser.add_argument('--sample_strategy', '-ss', type=str, default='rw')
parser.add_argument('--load_save', type=int, default=0)

parser.add_argument('--victim_model', '-v', type=str, default='WMFTrainer')
parser.add_argument('--fake_user_ratio', '-f', type=float, default=0.005)
parser.add_argument('--item', '-i', type=int, default=100)
parser.add_argument('--popularity_base', '-p', type=float, default=0.1)

args = parser.parse_args()
print("Argument: ")
print(args)

target_items = get_target_item_list(dataset=args.dataset)
p_item = args.item + len(target_items)
vict_args = get_args(args.dataset)
victim_model = args.victim_model
sub_vict_args = vict_args.vict_wmf_sgd
if victim_model != "WMFTrainer":
    print("don't find this model!")

t = time.time()
local_solution(dataset=args.dataset,
               sample_ratio=args.sample_ratio,
               sample_strategy=args.sample_strategy,
               args=vict_args,
               victim_model=victim_model,
               fake_user_ratio=args.fake_user_ratio,
               p_item=p_item,
               target_items=target_items,
               popularity_base=args.popularity_base,
               seed=0)
print('time cost: ', int(time.time() - t) / 60)
