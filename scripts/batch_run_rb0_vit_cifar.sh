# python main_parallel.py --config ./conf/rb_cifar10_vitsm_static_dirichlet.yaml
python main_parallel.py --config ./conf/rb_cifar10_vitsm_static_dirichlet.yaml --data_portion 0.5 --data_augmentation 2 --attack_type dba
python main_parallel.py --config ./conf/rb_cifar10_vitsm_static_dirichlet.yaml --data_portion 0.5 --data_augmentation 2 --attack_type dark
# python main_parallel.py --config ./conf/rb_cifar10_vitsm_static_dirichlet.yaml --data_portion 0.5 --data_augmentation 2 --gpu 1 --attack_type 3dfed
# python main_parallel.py --config ./conf/rb_cifar10_vitsm_static_dirichlet.yaml --data_portion 0.5 --data_augmentation 2 --gpu 1 --attack_type edge
# python main_parallel.py --config ./conf/rb_cifar10_vitsm_static_dirichlet.yaml --data_portion 0.5 --data_augmentation 2 --gpu 1 --attack_type invisible