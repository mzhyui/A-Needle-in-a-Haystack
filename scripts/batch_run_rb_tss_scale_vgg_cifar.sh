python main_parallel.py --config ./conf/rb_cifar10_vggsm_static_dirichlet.yaml --portion 0.15 --tss_hard_k 3 --tss_statistical_reject 1
python main_parallel.py --config ./conf/rb_cifar10_vggsm_static_dirichlet.yaml --portion 0.15 --tss_hard_k 3 --tss_statistical_reject 1 --attack_type blend
python main_parallel.py --config ./conf/rb_cifar10_vggsm_static_dirichlet.yaml --portion 0.15 --tss_hard_k 3 --tss_statistical_reject 1 --attack_type dba
python main_parallel.py --config ./conf/rb_cifar10_vggsm_static_dirichlet.yaml --portion 0.15 --tss_hard_k 3 --tss_statistical_reject 1 --attack_type dark --data_portion 0.7
python main_parallel.py --config ./conf/rb_cifar10_vggsm_static_dirichlet.yaml --portion 0.15 --tss_hard_k 3 --tss_statistical_reject 1 --attack_type 3dfed 
# python main_parallel.py --config ./conf/rb_cifar10_vggsm_static_dirichlet.yaml --portion 0.15 --tss_hard_k 3 --tss_statistical_reject 1 --attack_type edge
# python main_parallel.py --config ./conf/rb_cifar10_vggsm_static_dirichlet.yaml --portion 0.15 --tss_hard_k 3 --tss_statistical_reject 1 --attack_type invisible