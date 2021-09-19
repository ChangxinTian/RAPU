from trainers import WMFTrainer
from bunch import Bunch


def get_trainer(args,
                n_users,
                n_items,
                train_data,
                model_name="WMFTrainer",
                log_print=True):
    victim_args = None
    trainer = None
    if model_name == "WMFTrainer":
        victim_args = Bunch(args.vict_wmf_sgd)
        trainer = WMFTrainer(n_users=n_users,
                             n_items=n_items,
                             args=victim_args,
                             log_print=log_print)
    else:
        print("Don't find this model!")
    return trainer
