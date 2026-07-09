# python main_parallel.py --config ./conf/rb_cifar10_vggsm_static_dirichlet.yaml --debug --enable_rb 0
# python main_parallel.py --config ./conf/rb_cifar10_vggsm_static_dirichlet.yaml --debug --enable_rb 0 --attack_type dba
# python main_parallel.py --config ./conf/rb_cifar10_vggsm_static_dirichlet.yaml --debug --enable_rb 0 --attack_type dark
# python main_parallel.py --config ./conf/rb_cifar10_vggsm_static_dirichlet.yaml --debug --enable_rb 0 --attack_type 3dfed
python main_parallel.py --config ./conf/rb_cifar10_vggsm_static_dirichlet.yaml --debug --enable_rb 0 --attack_type invisible --bs 32 --local_bs 32