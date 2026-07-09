python main_parallel.py --config ./conf/rb_mnist_lenetsm_static_dirichlet.yaml --attack_type blend --data_portion 0.5 --local_ep_times 2
python main_parallel.py --config ./conf/rb_mnist_lenetsm_static_dirichlet.yaml --attack_type blend --data_portion 0.5 --local_ep_times 2 --tss_statistical_reject 1 --tss_layer_count 2
python main_parallel.py --config ./conf/rb_mnist_lenetsm_static_dirichlet.yaml --attack_type blend --data_portion 0.5 --local_ep_times 2 --mkrum
python main_parallel.py --config ./conf/rb_mnist_lenetsm_static_dirichlet.yaml --attack_type blend --data_portion 0.5 --local_ep_times 2 --rlr
python main_parallel.py --config ./conf/rb_mnist_lenetsm_static_dirichlet.yaml --attack_type blend --data_portion 0.5 --local_ep_times 2 --tracer
python main_parallel.py --config ./conf/rb_mnist_lenetsm_static_dirichlet.yaml --attack_type blend --data_portion 0.5 --local_ep_times 2 --flame
