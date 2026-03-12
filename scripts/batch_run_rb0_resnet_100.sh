python main_parallel.py --config ./conf/rb_100_resnet20sm_static_dirichlet.yaml --data_augmentation 2 
python main_parallel.py --config ./conf/rb_100_resnet20sm_static_dirichlet.yaml --data_augmentation 2 --attack_type blend
python main_parallel.py --config ./conf/rb_100_resnet20sm_static_dirichlet.yaml --data_augmentation 2 --attack_type dba
python main_parallel.py --config ./conf/rb_100_resnet20sm_static_dirichlet.yaml --data_augmentation 2 --attack_type dark --data_portion 0.7
python main_parallel.py --config ./conf/rb_100_resnet20sm_static_dirichlet.yaml --data_augmentation 2 --attack_type 3dfed 
python main_parallel.py --config ./conf/rb_100_resnet20sm_static_dirichlet.yaml --data_augmentation 2 --attack_type invisible --bs 32 --local_bs 32