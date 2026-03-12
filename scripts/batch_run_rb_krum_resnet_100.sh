python main_parallel.py --config ./conf/rb_100_resnet20sm_static_dirichlet.yaml --mkrum
# python main_parallel.py --config ./conf/rb_100_resnet20sm_static_dirichlet.yaml --mkrum --attack_type blend
python main_parallel.py --config ./conf/rb_100_resnet20sm_static_dirichlet.yaml --mkrum --attack_type dba
python main_parallel.py --config ./conf/rb_100_resnet20sm_static_dirichlet.yaml --mkrum --attack_type dark --data_portion 0.7
python main_parallel.py --config ./conf/rb_100_resnet20sm_static_dirichlet.yaml --mkrum --attack_type 3dfed
python main_parallel.py --config ./conf/rb_100_resnet20sm_static_dirichlet.yaml --mkrum --attack_type invisible --threads 4 --bs 32 --local_bs 32