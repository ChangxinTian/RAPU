import importlib
from bunch import Bunch


def get_args(dataset):
    if dataset == 'ml-100k':
        args_loc = 'args.baseline_ml100k_args'
    elif dataset == 'Video':
        args_loc = 'args.baseline_video_args'
    else:
        print("have no dataset ", dataset)
        args_loc = 'args.baseline_ml100k_args'
    args = importlib.import_module(args_loc)
    return args


def get_attack_args(dataset):
    gen_args = importlib.import_module("args.baseline_generate_attack_args")
    if dataset == 'ml-100k':
        attack_gen_args = Bunch(gen_args.attack_gen_args_ml100k_our)
    elif dataset == 'Video':
        attack_gen_args = Bunch(gen_args.attack_gen_args_video_our)
    else:
        print("have no dataset ", dataset)
    return attack_gen_args
