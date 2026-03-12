python main_parallel.py --config ./conf/rb_100_resnet20sm_static_dirichlet.yaml --portion 0.15 --tss_hard_k 4 --tss_statistical_reject 1
python main_parallel.py --config ./conf/rb_100_resnet20sm_static_dirichlet.yaml --portion 0.15 --tss_hard_k 4 --tss_statistical_reject 1 --attack_type blend
python main_parallel.py --config ./conf/rb_100_resnet20sm_static_dirichlet.yaml --portion 0.15 --tss_hard_k 4 --tss_statistical_reject 1 --attack_type dba
python main_parallel.py --config ./conf/rb_100_resnet20sm_static_dirichlet.yaml --portion 0.15 --tss_hard_k 4 --tss_statistical_reject 1 --attack_type dark --data_portion 0.7
python main_parallel.py --config ./conf/rb_100_resnet20sm_static_dirichlet.yaml --portion 0.15 --tss_hard_k 4 --tss_statistical_reject 1 --attack_type 3dfed 
# python main_parallel.py --config ./conf/rb_100_resnet20sm_static_dirichlet.yaml --portion 0.15 --tss_hard_k 4 --tss_statistical_reject 1 --attack_type invisible