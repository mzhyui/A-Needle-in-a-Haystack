# python main_parallel.py --config ./conf/rb_gtsrb_resnet20sm_static_dirichlet.yaml --mkrum
# python main_parallel.py --config ./conf/rb_gtsrb_resnet20sm_static_dirichlet.yaml --mkrum --attack_type dba
python main_parallel.py --config ./conf/rb_gtsrb_resnet20sm_static_dirichlet.yaml --mkrum --attack_type dark --portion 0.7
python main_parallel.py --config ./conf/rb_gtsrb_resnet20sm_static_dirichlet.yaml --mkrum --attack_type 3dfed
python main_parallel.py --config ./conf/rb_gtsrb_resnet20sm_static_dirichlet.yaml --mkrum --attack_type invisible