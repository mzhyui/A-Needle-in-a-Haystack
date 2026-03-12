python main_parallel.py --config ./conf/rb_gtsrb_vggsm_static_dirichlet.yaml --data_augmentation 1 --flame
python main_parallel.py --config ./conf/rb_gtsrb_vggsm_static_dirichlet.yaml --data_augmentation 1 --flame --attack_type blend
python main_parallel.py --config ./conf/rb_gtsrb_vggsm_static_dirichlet.yaml --data_augmentation 1 --flame --attack_type dba
python main_parallel.py --config ./conf/rb_gtsrb_vggsm_static_dirichlet.yaml --data_augmentation 1 --flame --attack_type dark --data_portion 0.7
python main_parallel.py --config ./conf/rb_gtsrb_vggsm_static_dirichlet.yaml --data_augmentation 1 --flame --attack_type 3dfed
# python main_parallel.py --config ./conf/rb_gtsrb_vggsm_static_dirichlet.yaml --flame --attack_type invisible --bs 32 --local_bs 32 --parallel 0 --threads 4
