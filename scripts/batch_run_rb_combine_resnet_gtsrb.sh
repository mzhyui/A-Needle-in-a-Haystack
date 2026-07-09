# python main_parallel.py --config ./conf/rb_gtsrb_resnet20sm_static_dirichlet.yaml --mkrum
# python main_parallel.py --config ./conf/rb_gtsrb_resnet20sm_static_dirichlet.yaml --mkrum --attack_type dba
# python main_parallel.py --config ./conf/rb_gtsrb_resnet20sm_static_dirichlet.yaml --mkrum --attack_type dark --portion 0.7
# python main_parallel.py --config ./conf/rb_gtsrb_resnet20sm_static_dirichlet.yaml --mkrum --attack_type 3dfed
# python main_parallel.py --config ./conf/rb_gtsrb_resnet20sm_static_dirichlet.yaml --mkrum --attack_type invisible

python main_parallel.py --config ./conf/rb_gtsrb_resnet20sm_static_dirichlet.yaml --attack_type blend --tss_statistical_reject 1
python main_parallel.py --config ./conf/rb_gtsrb_resnet20sm_static_dirichlet.yaml --attack_type blend --mkrum
python main_parallel.py --config ./conf/rb_gtsrb_resnet20sm_static_dirichlet.yaml --attack_type blend --rlr
python main_parallel.py --config ./conf/rb_gtsrb_resnet20sm_static_dirichlet.yaml --attack_type blend --tracer
# python main_parallel.py --config ./conf/rb_gtsrb_resnet20sm_static_dirichlet.yaml --attack_type blend --flame