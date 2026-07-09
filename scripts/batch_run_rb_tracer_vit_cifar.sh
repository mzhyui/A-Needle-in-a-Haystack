python main_parallel.py --config ./conf/rb_cifar10_vitsm_static_dirichlet.yaml --tracer --data_portion 0.5 
python main_parallel.py --config ./conf/rb_cifar10_vitsm_static_dirichlet.yaml --tracer --data_portion 0.5 --attack_type blend
python main_parallel.py --config ./conf/rb_cifar10_vitsm_static_dirichlet.yaml --tracer --data_portion 0.7 --attack_type dba
python main_parallel.py --config ./conf/rb_cifar10_vitsm_static_dirichlet.yaml --tracer --data_portion 0.7 --attack_type dark
python main_parallel.py --config ./conf/rb_cifar10_vitsm_static_dirichlet.yaml --tracer --data_portion 0.5 --attack_type 3dfed 
# python main_parallel.py --config ./conf/rb_cifar10_vitsm_static_dirichlet.yaml --tracer --data_portion 0.5 --attack_type edge
# python main_parallel.py --config ./conf/rb_cifar10_vitsm_static_dirichlet.yaml --tracer --data_portion 0.5 --attack_type invisible