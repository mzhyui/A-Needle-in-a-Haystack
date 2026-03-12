# FL_box_dev
FL environments with attack and defense implementation.

Forked from [LG-FedAvg](https://github.com/pliang279/LG-FedAvg)

# Intergrated Functions
>   [DBA](https://github.com/AI-secure/DBA) \
    [Lipschitzness Channel Distance](https://github.com/rkteddychannel-Lipschitzness-based-pruning.git) \
    [FedSAM](https://github.com/debcaldarola/fedsam.git) \
    [Continual Learning](https://github.com/ZixuanKe/PyContinual) \
    [3Dfed](https://github.com/haoyangliASTAPLE/3DFed) \
    [darkfed](https://github.com/hustweiwan/DarkFed) \
    [flame-unofficial](https://github.dev/zhmzm/FLAME) \
    [Neural Cleanse](https://github.com/bolunwang/backdoor) \ 
    [Neural Cleanse Implementation](https://github.com/tonggege001/MyNeuralCleanse)


# TODO

## attack
- [ ] TODO 2025-01-19 git.V.29c9f: decoy attack?
- [ ] TODO 2025-02-21 git.V.894ba: pattern value outside (0,1)?
    - [ ] **TODO 2025-02-21 git.V.894ba: local model asr 0.00?**
- [ ] TODO 2025-02-19 git.V.894ba: static attack backdoor pattern method leads to entangled model tss
- [ ] TODO 2025-02-22 git.V.894ba: Gaussian noise attack
- [ ] TODO 2025-02-27 git.V.894ba: separate normal loss and backdoor loss
- [ ] **TODO 2025-03-04 git.V.e68d2: global ASR gain dropped**
      - [x] TODO 2025-03-13 git.V.96ca0: share static model between objects
      - [ ] **TODO 2025-05-31 git.V.5d39d: parallel triggernet update**
- [ ] TODO 2025-03-13 git.V.81bed: A3FL / IBA / DARK
    - [ ] TODO 2025-02-27 git.V.36afa: fix dark shadow dataset.
    - [ ] **TODO 2025-03-15 git.V.1d64c: softlabel matching for DarkFed**
    - [x] TODO 2025-03-04 git.V.e68d2: 3dfed cifar10 test
    - [ ] TODO 2025-05-15 git.V.6beb3: 3dfed vit test
- [ ] TODO 2025-04-10 git.V.2a1e0: backdoor performance poor on vit
- [ ] TODO 2025-05-06 git.V.5bac8: use backdoor dataloader
- [ ] TODO 2025-05-21 git.V.e294a: multiple attack different attackers?
- [ ] TODO 2025-06-17 git.V.c4b43: DBA poor on vitsm
- [ ] TODO 2025-05-31 git.V.5d39d: invisible fix
  - [ ] TODO 2025-03-10 git.V.42bf4: invisible cannot adapt 1 channel input
  - [ ] TODO 2025-03-12 git.V.81bed: attacker update network in invisible
      - [ ] **TODO 2025-03-12 git.V.81bed: don't use parallel with invisible**
  - [ ] TODO 2025-06-18 git.V.7a24e: invisible vram issue
  - [ ] invisible maintask acc 10%
- [ ] edge attack dataset mixture
- [ ] TODO 2025-06-30 git.V.cf31a: edge attack asr cannot maintain
  - [ ] TODO 2025-06-30 git.V.cf31a: mix dataset
- [ ] TODO 2025-06-30 git.V.cf31a: mirage attack
  - [ ] TODO 2025-09-08 git.V.0e0a3: asr low
  - [ ] TODO 2025-09-08 git.V.0e0a3: multi-attacker trigger searching and updating

## defense
- [x] TODO 2025-03-07 git.V.0f748: hard reject / soft reject (weight) / selective reject (layer)
    - [x] TODO 2025-03-10 git.V.3fa0b: layer router?
        - [x] **TODO 2025-03-13 git.V.81bed: soft reject**
            - [ ] TODO 2025-03-14 git.V.a4ecc: fail to aggregate with soft weight
        - [ ] TODO 2025-03-13 git.V.81bed: network architecture search / meta learning
            - [ ] TODO 2025-03-20 git.V.17073: router select layers / neurons to aggregate
        - [ ] use router to assign layer weight, based on statistical analysis (weight size, etc)
- [ ] TODO 2025-03-10 git.V.3fa0b: Frequency. EG, defend at each 5 rounds, what if attack happens within 5 rounds and evade detection?
- [ ] TODO 2025-03-10 git.V.3fa0b: Clean ASR
- [ ] TODO 2025-03-23 git.V.fa99d: distance between attackers? use softmax function?
- [x] TODO 2025-04-24 git.V.ea4c0: TSS replace validation set with noise
    - [ ] TODO 2025-05-08 git.V.9ee3d: explore why gaussian noise works.
    - [x] TODO 2025-05-12 git.V.6b6d3: gaussian noise distribution parameter based on task (label)
        - [ ] TODO 2025-05-12 git.V.6b6d3: double check function `SynthesizeDataset`
- [ ] TODO 2025-05-08 git.V.9ee3d: apply gmm historical ban
- [x] TODO 2025-05-12 git.V.6b6d3: visualize tss layer differences
  - [ ] TODO 2025-05-14 git.V.6beb3: tss matrices sort by mean value visualization
- [ ] TODO 2025-05-18 git.V.3b625: for those methods which regulates on l2 norm, how to detect
- [x] TODO 2025-05-18 git.V.3b625: layer-wise cluster, then aggregrate
  - [ ] TODO 2025-05-25 git.V.c24c0: still needs detection acc high
- [x] TODO 2025-05-26 git.V.dc4d1: normalize weight before tss?
  - [ ] TODO 2025-06-04 git.V.5d39d: normalize only the delta part
- [ ] TODO 2025-05-28 git.V.c0223: tss between attack group is relatively low
- [ ] TODO 2025-06-17 git.V.c4b43: rlr maintask drop
- [ ] TODO 2025-06-17 git.V.c4b43: clipping asr high
- [ ] TODO 2025-06-17 git.V.c6de3: fl tracer flot error
- [x] TODO 2025-09-08 git.V.0e0a3: trigger inversion based on masked network
  - [ ] TODO 2025-09-17 git.V.6d315: multi-model adaption
  - [ ] TODO 2025-09-17 git.V.6d315: apply trigger activation mask

## Evaluation
- [] TODO 2025-11-13 git.V.b482e: Using local training gss mask, compare the importance mask with eval-masking

## dataset
- [x] TODO 2025-02-17 git.V.ca07f: Use dirichlet distribution to replace non-iid
    - [ ] TODO 2025-02-17 git.V.ca07f: align user's training dataset distribution and test dataset distribution
- [ ] TODO 2025-03-05 git.V.6ef8a: gtsrb / celeba
    - [x] TODO 2025-03-07 git.V.87012: impact of large num_classes
        - [ ] TODO 2025-04-08 git.V.059d4: gtsrb worked cifar100 not tested
    - [x] TODO 2025-03-10 git.V.42bf4: gtsrb cannot defend
- [x] TODO 2025-03-10 git.V.83183: validation dataset size?
    - [ ] TODO 2025-03-06 git.V.33573: use generated testset
    - [x] TODO 2025-03-13 git.V.81bed: shrink validation dataset size
    - [ ] TODO 2025-04-08 git.V.059d4: use lower frac
- [ ] TODO 2025-05-24 git.V.008d6: local client's data augmentation
  - [ ] TODO 2025-05-26 git.V.dc4d1: increase epochs

## model
- [ ] TODO 2025-04-03 git.V.52fcd: global convergence failed
- [x] TODO 2025-04-08 git.V.059d4: vit sm replacement
    - [ ] TODO 2025-04-09 git.V.059d4: vit sm defense
    - [ ] TODO 2025-04-09 git.V.2a1e0: vit attention head sm substitution


## performance
- [ ] **TODO 2025-03-13 git.V.81bed: vram usage fix**
    - [ ] **TODO 2025-03-19 git.V.0d918: vgg vram cannot be fully deallocated**
    - [x] TODO 2025-03-25 git.V.52fcd: multi-gpu optimization
        - [ ] **TODO 2025-04-08 git.V.059d4: gpu 0 consumes vram after selecting gpu 1 as worker**
        - [ ] fork threads are taking vram
    - [ ] TODO 2025-10-19 git.V.cb1f4: check `total allocated memory` extreme usage
      - [ ] TODO 2025-10-19 git.V.cb1f4: non-parallel
      - [ ] TODO 2025-10-19 git.V.cb1f4: tss caused vram error
        - tss introduces 10 times vram usage, with resnet20sm
    - [ ] TODO 2025-10-19 git.V.cb1f4: initialize updaters when program begins
- [ ] TODO 2025-04-09 git.V.059d4: vit tss calculation bottleneck
- [ ] TODO 2024-12-13 git.V.9ea50:
    - [ ] [W1213 21:05:40.798672132 CudaIPCTypes.cpp:96] Producer process tried to deallocate over 1000 memory blocks referred by consumer processes. Deallocation might be significantly slowed down. We assume it will never going to be the case, but if it is, please file bug to https://github.com/pytorch/pytorch
- [ ] TODO 2025-05-25 git.V.c24c0: non-parallel method raise fork error
- [ ] mix of SGD and ADAM cause byzantine failuer in global model.
- [ ] **TODO 2025-07-27 git.V.81a4d: load base model weight into sm weight and eval**

## misc
- [ ] TODO 2024-12-11 git.V.2a916: save local model function needs fix (using % save is deprecated)
- [ ] TODO 2024-12-11 git.V.2a916: change options from store_true to int values
- [ ] TODO 2025-01-21 git.V.b9b8c: in non-iid setting, user upload their label list?
- [ ] TODO 2025-03-19 git.V.0d567: plot data distributions for each user
- [ ] **TODO 2025-04-08 git.V.059d4: function test after code major rewrite**
- [ ] TODO 2025-04-14 git.V.90ac3: fan function not supported in pytorch >= 2.2
- [ ] TODO 2025-04-25 git.V.ea4c0: dedicated cuda device id not following the nvidia-smi shown id
- [ ] TODO 2025-05-10 git.V.3c674: orthogonality analysis: the initial attack is most orthogonal to the global model?
- [ ] record the current git version into logfile
- [ ] TODO 2025-05-18 git.V.ef483: local ep >= 5?
- [ ] TODO 2025-05-21 git.V.e294a: continue from
  - [ ] TODO 2025-05-21 git.V.e294a: load fed / user dict / last epoch in folder
  - [ ] TODO 2025-05-21 git.V.e294a: continue log
- [ ] threads -> workers
- [ ] TODO 2025-09-05 git.V.300b8: split parameters from args

## TEST TARGETS AFTER MERGING
- [ ] ASR
- [ ] MAINTASK ACC
- [ ] LOCAL ACC

# DONE
- [x] TODO 2024-12-06 git.V.f73ac: fix non-iid implementation
- [x] TODO 2024-12-07 git.V.f73ac: parallel training
    - [x] TODO 2024-12-07 git.V.f73ac: slow
- [x] TODO 2024-12-30 git.V.bb195: validate orthogonality with different label as a task
- [x] TODO 2024-12-30 git.V.bb195: vgg vram issue
    - [x] TODO 2024-12-30 git.V.e0216: to cpu?
- [x] TODO 2025-01-21 git.V.b9b8c: attack on fmnist
- [x] TODO 2025-01-21 git.V.b9b8c: iid?
- [x] TODO 2025-02-18 git.V.ca07f: TSS calculation efficiency
    - [x] TODO 2025-02-22 git.V.894ba: Optimize TSS calculation
    - [x] TODO 2025-01-17 git.V.29c9f: TSS angle compare multi-processing
- [x] TODO 2025-02-19 git.V.894ba: frac < 1 leads to missing key in TSS dict
- [x] TODO 2025-02-21 git.V.894ba: fix 1 channel backdoor pattern application
- [x] TODO 2025-03-17 git.V.0d918: use larger model
    - [x] TODO 2025-03-18 git.V.0d918: still acc 10%
- [x] TODO 2025-02-21 git.V.894ba: show trigger
- [x] TODO 2025-03-12 git.V.4498c: fmnist ASR raised without attack
    - [x] TODO 2025-03-19 git.V.972a0: used wrong normalizer
- [x] TODO 2025-03-12 git.V.4498c: AugmentedDataset rework
- [x] TODO 2025-03-12 git.V.a09b2: load user dict when load fed
- [x] TODO 2025-03-13 git.V.81bed: user dicts initialization has impacts on TSS detection
- [x] TODO 2025-03-23 git.V.52fcd: nan in TSS similarity
- [x] TODO 2025-05-21 git.V.e294a: edge attack


# Operation Commands
`python main_parallel.py --config ./conf/rb_cifar10_resnet20sm_static_dirichlet.yaml --attack_type dba --debug --tss_threshold 0.7` # TODO 2025-06-17 git.V.c4b43: 

`bash ./scripts/get_base.sh -f ./scripts/batch_run_rb0_resnet_gtsrb.sh` #
