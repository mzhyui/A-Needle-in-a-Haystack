# python main_parallel.py --config ./conf/rb_cifar10_vggsm_static_dirichlet.yaml --data_portion 0.4 --debug --tracer
# python main_parallel.py --config ./conf/rb_cifar10_vggsm_static_dirichlet.yaml --data_portion 0.4 --debug --tracer --attack_type dba
# python main_parallel.py --config ./conf/rb_cifar10_vggsm_static_dirichlet.yaml --data_portion 0.7 --debug --tracer --attack_type dark
python main_parallel.py --config ./conf/rb_cifar10_vggsm_static_dirichlet.yaml --data_portion 0.3 --gpu 1 --tracer --attack_type 3dfed
python main_parallel.py --config ./conf/rb_cifar10_vggsm_static_dirichlet.yaml --data_portion 0.3 --gpu 1 --tracer --attack_type invisible --bs 32 --local_bs 32
python main_parallel.py --config ./conf/rb_cifar10_vggsm_static_dirichlet.yaml --data_portion 0.3 --gpu 1 --tracer --attack_type edge