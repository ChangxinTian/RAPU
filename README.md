# Recommendation Attack for Partial and Perturbed Data (RAPU)

This is our Pytorch implementation for the paper:

> Hengtong Zhang*, Changxin Tian*, Yaliang Li, Lu Su, Nan Yang, Wayne Xin Zhao, and Jing Gao (2021). "Data Poisoning Attack against Recommender System Using Incomplete and Perturbed Data." In KDD 2021.

## Introduction
Recent studies reveal that recommender systems are vulnerable to data poisoning attack due to their openness nature. In data poisoning attack, the attacker typically recruits a group of controlled
users to inject well-crafted user-item interaction data into the recommendation model’s training set to modify the model parameters as desired. Thus, existing attack approaches usually require full access to the training data to infer items’ characteristics and craft
the fake interactions for controlled users. However, such attack approaches may not be feasible in practice due to the attacker’s limited data collection capability and the restricted access to the training data, which sometimes are even perturbed by the privacy
preserving mechanism of the service providers. Such design-reality gap may cause failure of attacks. In this paper, we fill the gap by proposing two novel adversarial attack approaches to handle the incompleteness and perturbations in user-item interaction data.
First, we propose a bi-level optimization framework that incorporates a probabilistic generative model to find the users and items whose interaction data are sufficient and have not been significantly perturbed, and leverage these users and items’ data to craft fake
user-item interactions. Moreover, we reverse the learning process of recommendation models and develop a simple yet effective approach that can incorporate context-specific heuristic rules to handle data incompleteness and perturbations. Extensive experiments
on two datasets against three representative recommendation models show that the proposed approaches can achieve better attack performance than existing approaches.

## Requirements:
* Python=3.7.7
* PyTorch=1.7.0
* pandas=1.1.0
* numpy=1.19.2

## Training:
run RAPU_G
```
python -u main_G.py -v WMFTrainer -d ml-100k -i 100
```

run RAPU_R
```
python -u main_R.py -v WMFTrainer -d ml-100k -i 100
```

## Reference:
Any scientific publications that use our codes should cite the following paper as the reference:

 ```
 @inproceedings{Zhang-KDD-2021,
     title = "Data Poisoning Attack against Recommender System Using Incomplete and Perturbed Data",
     author = {Hengtong Zhang and
     		  Changxin Tian and
     		  Yaliang Li and
     		  Lu Su and
     		  Nan Yang and
     		  Wayne Xin Zhao and
     		  Jing Gao},
     booktitle = {{KDD}},
     year = {2021},
 }
 ```

If you have any questions for our paper or codes, please send an email to tianchangxin@ruc.edu.cn.
