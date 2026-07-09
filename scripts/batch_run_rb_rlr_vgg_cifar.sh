python main_parallel.py --config ./conf/rb_cifar10_vggsm_static_dirichlet.yaml --rlr
python main_parallel.py --config ./conf/rb_cifar10_vggsm_static_dirichlet.yaml --rlr --attack_type dba
python main_parallel.py --config ./conf/rb_cifar10_vggsm_static_dirichlet.yaml --rlr --attack_type dark --data_portion 0.7
python main_parallel.py --config ./conf/rb_cifar10_vggsm_static_dirichlet.yaml --rlr --attack_type 3dfed
python main_parallel.py --config ./conf/rb_cifar10_vggsm_static_dirichlet.yaml --rlr --attack_type invisible --bs 32 --local_bs 32 --parallel 0 --threads 4
python main_parallel.py --config ./conf/rb_cifar10_vggsm_static_dirichlet.yaml --rlr --attack_type edge
# python main_parallel.py --config ./conf/rb_gtsrb_vggsm_static_dirichlet.yaml --rlr
# python main_parallel.py --config ./conf/rb_gtsrb_vggsm_static_dirichlet.yaml --rlr --attack_type dba
# python main_parallel.py --config ./conf/rb_gtsrb_vggsm_static_dirichlet.yaml --rlr --attack_type dark --data_portion 0.7
# python main_parallel.py --config ./conf/rb_gtsrb_vggsm_static_dirichlet.yaml --rlr --attack_type 3dfed
# python main_parallel.py --config ./conf/rb_gtsrb_vggsm_static_dirichlet.yaml --rlr --attack_type invisible --bs 32 --local_bs 32 --parallel 0 --threads 4
