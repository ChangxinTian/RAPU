from utils.metrics import *
from trainers import *
from trainers.losses import *

use_cuda = True  # If using GPU or CPU
seed = 1234  # Random seed
metrics = [PrecisionRecall(k=[10, 50]), NormalizedDCG(k=[10, 50])]

shared_params = {
    "use_cuda": use_cuda,
    "metrics": metrics,
    "seed": seed,
    "output_dir": "./checkpoint/",
}

vict_wmf_sgd = {
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
