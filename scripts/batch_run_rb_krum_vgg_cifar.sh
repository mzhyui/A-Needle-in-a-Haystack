python main_parallel.py --config ./conf/rb_cifar10_vggsm_static_dirichlet.yaml --mkrum --gpu 1
python main_parallel.py --config ./conf/rb_cifar10_vggsm_static_dirichlet.yaml --mkrum --gpu 1 --attack_type dba
python main_parallel.py --config ./conf/rb_cifar10_vggsm_static_dirichlet.yaml --mkrum --gpu 1 --attack_type dark --portion 0.7
python main_parallel.py --config ./conf/rb_cifar10_vggsm_static_dirichlet.yaml --mkrum --gpu 1 --attack_type 3dfed
python main_parallel.py --config ./conf/rb_cifar10_vggsm_static_dirichlet.yaml --mkrum --gpu 1 --attack_type invisible --parallel 0 --threads 4 --bs 32 --local_bs 32
python main_parallel.py --config ./conf/rb_cifar10_vggsm_static_dirichlet.yaml --mkrum --gpu 1 --attack_type edge