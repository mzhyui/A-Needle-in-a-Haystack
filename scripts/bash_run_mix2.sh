python main_parallel.py --config ./conf/rb_cifar10_resnet20sm_static_dirichlet.yaml  --attack_type dark --rlr
python main_parallel.py --config ./conf/rb_gtsrb_resnet20sm_static_dirichlet.yaml --mkrum
python main_parallel.py --config ./conf/rb_gtsrb_resnet20sm_static_dirichlet.yaml --attack_type dba --mkrum
python main_parallel.py --config ./conf/rb_gtsrb_resnet20sm_static_dirichlet.yaml --attack_type dark --mkrum
python main_parallel.py --config ./conf/rb_gtsrb_resnet20sm_static_dirichlet.yaml --attack_type dark --rlr
python main_parallel.py --config ./conf/rb_cifar10_resnet20sm_static_dirichlet.yaml  --attack_type edge --mkrum --load_fed fl_save/release/cifar10/dirichlet0.5/resnet20sm_num-20_C-1/edge/lr0.01ep3/Krum_06-27--07-26-29/fed/attack_portion0.1_model_40.pt --load_begin_epoch 40 --load_user_dict fl_save/release/cifar10/dirichlet0.5/resnet20sm_num-20_C-1/edge/lr0.01ep3/Krum_06-27--07-26-29/dict_users.pkl