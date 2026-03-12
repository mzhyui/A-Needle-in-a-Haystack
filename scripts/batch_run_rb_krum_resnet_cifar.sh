python main_parallel.py --config ./conf/rb_cifar10_resnet20sm_static_dirichlet.yaml --gpu 2 --mkrum
python main_parallel.py --config ./conf/rb_cifar10_resnet20sm_static_dirichlet.yaml --gpu 2 --mkrum --attack_type dba
python main_parallel.py --config ./conf/rb_cifar10_resnet20sm_static_dirichlet.yaml --gpu 2 --mkrum --attack_type dark --portion 0.7
python main_parallel.py --config ./conf/rb_cifar10_resnet20sm_static_dirichlet.yaml --gpu 2 --mkrum --attack_type 3dfed
python main_parallel.py --config ./conf/rb_cifar10_resnet20sm_static_dirichlet.yaml --gpu 2 --mkrum --attack_type invisible --parallel 0 --threads 4 --bs 32 --local_bs 32