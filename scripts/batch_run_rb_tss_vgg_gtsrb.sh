python main_parallel.py --config ./conf/rb_gtsrb_vggsm_static_dirichlet.yaml --tss_statistical_reject 1 --data_augmentation 2
python main_parallel.py --config ./conf/rb_gtsrb_vggsm_static_dirichlet.yaml --tss_statistical_reject 1 --data_augmentation 2 --attack_type dark --data_portion 0.7
python main_parallel.py --config ./conf/rb_gtsrb_vggsm_static_dirichlet.yaml --tss_statistical_reject 1 --data_augmentation 2 --attack_type 3dfed 
python main_parallel.py --config ./conf/rb_gtsrb_vggsm_static_dirichlet.yaml --tss_statistical_reject 1 --data_augmentation 2 --attack_type edge
python main_parallel.py --config ./conf/rb_gtsrb_vggsm_static_dirichlet.yaml --tss_statistical_reject 1 --data_augmentation 2 --attack_type invisible