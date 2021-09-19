from utils.metrics import *
from trainers import *
from trainers.losses import *

use_cuda = True  # If using GPU or CPU
seed = 1234  # Random seed
metrics = [PrecisionRecall(k=[50]), NormalizedDCG(k=[50])]

shared_params = {
    "use_cuda": use_cuda,
    "metrics": metrics,
    "seed": seed,
    "output_dir": "./outputs/",
}

# #####################################    For ML-100k.    ######################################
sur_wmf_sgd = {
    **shared_params, "epochs": 201,
    "lr": 1e-2,
    "l2": 1e-5,
    "save_feq": 10,
    "batch_size": 2048,
    "valid_batch_size": 512,
    "model": {
        "trainer_class": WMFTrainer,
        "model_name": "WeightedMF-sgd",
        "hidden_dims": [64],
        "weight_alpha": 20,
        "optim_method": "sgd"
    }
}

attack_gen_args_ml100k_our = {
    **shared_params,
    "dataset": "ml-100k",
    "attack_type": "adversarial",
    "n_target_items": 5,

    # Args for adversarial training.
    "n_fakes": 0.01,
    "adv_epochs": 30,
    "unroll_steps": 20,
    "adv_lr": 1,
    "adv_momentum": 0.95,
    "proj_threshold": 0.0,
    "click_targets": True,

    # Args for surrogate model.
    "surrogate": sur_wmf_sgd
}

# ###################################### For Video. ######################################

sur_wmf_sgd_video = {
    **shared_params, "epochs": 101,
    "lr": 2e-2,
    "l2": 1e-5,
    "save_feq": 10,
    "batch_size": 2048,
    "valid_batch_size": 512,
    "model": {
        "trainer_class": WMFTrainer,
        "model_name": "WeightedMF-sgd",
        "hidden_dims": [32],
        "weight_alpha": 20,
        "optim_method": "sgd"
    }
}

attack_gen_args_video_our = {
    **shared_params,
    "dataset": "Video",
    "attack_type": "adversarial",
    "n_target_items": 5,

    # Args for adversarial training.
    "n_fakes": 0.01,
    "adv_epochs": 30,
    "unroll_steps": 20,
    "adv_lr": 1,
    "adv_momentum": 0.95,
    "proj_threshold": 0.0,
    "click_targets": True,

    # Args for surrogate model.
    "surrogate": sur_wmf_sgd_video
}
