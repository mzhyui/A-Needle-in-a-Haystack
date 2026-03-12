# python main_parallel.py --config ./conf/rb_gtsrb_resnet20sm_static_dirichlet.yaml --mkrum
# python main_parallel.py --config ./conf/rb_gtsrb_resnet20sm_static_dirichlet.yaml --mkrum --attack_type dba
# python main_parallel.py --config ./conf/rb_gtsrb_resnet20sm_static_dirichlet.yaml --mkrum --attack_type dark --portion 0.7
# python main_parallel.py --config ./conf/rb_gtsrb_resnet20sm_static_dirichlet.yaml --mkrum --attack_type 3dfed
# python main_parallel.py --config ./conf/rb_gtsrb_resnet20sm_static_dirichlet.yaml --mkrum --attack_type invisible

python main_parallel.py --config ./conf/rb_cifar10_vggsm_static_dirichlet.yaml --attack_type blend
python main_parallel.py --config ./conf/rb_gtsrb_vggsm_static_dirichlet.yaml --attack_type blend
python main_parallel.py --config ./conf/rb_cifar10_resnet20sm_static_dirichlet.yaml --attack_type blend
python main_parallel.py --config ./conf/rb_gtsrb_resnet20sm_static_dirichlet.yaml --attack_type blend
python main_parallel.py --config ./conf/rb_100_resnet20sm_static_dirichlet.yaml --attack_type blend
python main_parallel.py --config ./conf/rb_cifar10_vitsm_static_dirichlet.yaml --attack_type blend