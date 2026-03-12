# python main_parallel.py --config ./conf/rb_cifar10_resnet20sm_static_dirichlet.yaml --data_augmentation 2 --enable_rb 0 --gpu 2
# python main_parallel.py --config ./conf/rb_cifar10_resnet20sm_static_dirichlet.yaml --data_augmentation 2 --enable_rb 0 --gpu 2 --attack_type dba
python main_parallel.py --config ./conf/rb_cifar10_resnet20sm_static_dirichlet.yaml --data_augmentation 2 --enable_rb 0 --gpu 2 --attack_type dark --data_portion 0.7
python main_parallel.py --config ./conf/rb_cifar10_resnet20sm_static_dirichlet.yaml --data_augmentation 2 --enable_rb 0 --gpu 2 --attack_type 3dfed
python main_parallel.py --config ./conf/rb_cifar10_resnet20sm_static_dirichlet.yaml --data_augmentation 2 --enable_rb 0 --gpu 2 --attack_type invisible
python main_parallel.py --config ./conf/rb_cifar10_resnet20sm_static_dirichlet.yaml --data_augmentation 2 --enable_rb 0 --gpu 2 --attack_type edge