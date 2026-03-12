# python main_parallel.py --config ./conf/rb_gtsrb_vggsm_static_dirichlet.yaml --debug --mkrum 
# python main_parallel.py --config ./conf/rb_gtsrb_vggsm_static_dirichlet.yaml --debug --mkrum  --attack_type dba
# python main_parallel.py --config ./conf/rb_gtsrb_vggsm_static_dirichlet.yaml --debug --mkrum  --attack_type dark --data_portion 0.7
# python main_parallel.py --config ./conf/rb_gtsrb_vggsm_static_dirichlet.yaml --debug --mkrum  --attack_type 3dfed
python main_parallel.py --config ./conf/rb_gtsrb_vggsm_static_dirichlet.yaml --debug --mkrum  --attack_type invisible --bs 32 --local_bs 32