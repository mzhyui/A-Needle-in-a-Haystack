# python main_parallel.py --config ./conf/rb_mnist_lenetsm_static_dirichlet.yaml --flame
python main_parallel.py --config ./conf/rb_mnist_lenetsm_static_dirichlet.yaml --flame --attack_type blend
# python main_parallel.py --config ./conf/rb_mnist_lenetsm_static_dirichlet.yaml --flame --attack_type dba
# python main_parallel.py --config ./conf/rb_mnist_lenetsm_static_dirichlet.yaml --flame --attack_type dark --data_portion 0.7
python main_parallel.py --config ./conf/rb_mnist_lenetsm_static_dirichlet.yaml --flame --attack_type 3dfed