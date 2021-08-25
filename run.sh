conda activate py3.6
cd /mnt/data/shared/xyxu/projects/my-DML-projects/GDML-20210722
CUDA_VISIBLE_DEVICES=2 python main.py  --eval_every 1  --savename 'recover_best'  \
--dataset 'cub200' --arch 'resnet50_frozen_normalize' --batchminer_name 'distance'  --n_epochs  100 --BS 112 \
--loss_emb_name 'Margin'  --ft_params_lr 0.00001 --new_params_lr 0.00001 --decay 0.003  --tau 55 --gamma 0.3 \
--group_name 'LearnQ_fen' --k 3  --loss_div_name 'DivBD' --lam_div 0.02 \
--seed 4  --eval_begin 49 --reduction 8 \
--loss_margin_beta_lr 0.00005 \
--que_decay 0.001  --key_val_decay 0.003


conda activate py3.6
cd /mnt/data/shared/xyxu/projects/my-DML-projects/GDML-20210722
CUDA_VISIBLE_DEVICES=1 python main.py dataset 'online_products' \
group_name 'LearnQ_fen' k 3 reduction 8 decay 0.0005 \
key_val_decay 0.0005 que_decay 0.0001 eval_begin -1 eval_every 2 \
ft_params_lr 9e-6 new_params_lr 9e-6 loss_margin_beta_lr 1e-5 \
tau '[50, 1000]'




conda activate py3.6
cd /mnt/data/shared/xyxu/projects/my-DML-projects/GDML-20210722
CUDA_VISIBLE_DEVICES=9 python main.py dataset 'in_shop'

