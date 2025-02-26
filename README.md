# CLIP-LORA

This project is cloned from https://github.com/MaxZanella/CLIP-LoRA.

## Reproduce

CUDA_VISIBLE_DEVICES=5 python main.py --root_path /home/shenxi/Datasets --dataset food101 --seed 42 >> log/run_food_ema42.log

CUDA_VISIBLE_DEVICES=5 python main.py --root_path /home/shenxi/Datasets --dataset stanford_cars --seed 42 >> log/run_car_42.log

## Remove LoRA on q/k/v

CUDA_VISIBLE_DEVICES=5 python main.py --root_path /home/shenxi/Datasets --dataset stanford_cars --seed 42 --params k v >> log/run_car_rm_q_42.log

CUDA_VISIBLE_DEVICES=5 python main.py --root_path /home/shenxi/Datasets --dataset stanford_cars --seed 42 --params q v >> log/run_car_rm_k_42.log

CUDA_VISIBLE_DEVICES=8 python main.py --root_path /home/shenxi/Datasets --dataset stanford_cars --seed 42 --params q k >> log/run_car_rm_v_42.log

CUDA_VISIBLE_DEVICES=4 python main.py --root_path /home/shenxi/Datasets --dataset imagenet --seed 42 >> log/run_Inet_42.log

CUDA_VISIBLE_DEVICES=1 python main.py --root_path /home/shenxi/Datasets --dataset imagenet --seed 42 --params k v >> log/run_Inet_rm_q_42.log

CUDA_VISIBLE_DEVICES=2 python main.py --root_path /home/shenxi/Datasets --dataset imagenet --seed 42 --params q v >> log/run_Inet_rm_k_42.log

CUDA_VISIBLE_DEVICES=3 python main.py --root_path /home/shenxi/Datasets --dataset imagenet --seed 42 --params q k >> log/run_Inet_rm_v_42.log

CUDA_VISIBLE_DEVICES=0 python main.py --root_path /home/shenxi/Datasets --dataset imagenet --seed 42 --params v >> log/run_Inet_rm_qk_42.log

CUDA_VISIBLE_DEVICES=0 python main.py --root_path /datassd --dataset imagenet --seed 42 --params q k v o --batch_size 16 >> log/run_Inet_addo_42.log

## Remove LoRA on text encoder

CUDA_VISIBLE_DEVICES=0 python main.py --root_path /home/shenxi/Datasets --dataset stanford_cars --seed 42 --encoder vision >> log/run_car_rm_text_42.log

CUDA_VISIBLE_DEVICES=2 python main.py --root_path /home/shenxi/Datasets --dataset imagenet --seed 42 --encoder vision >> log/run_Inet_rm_text_42.log

## Remove LoRA on shallow layer

CUDA_VISIBLE_DEVICES=2 python main.py --root_path /home/shenxi/Datasets --dataset imagenet --seed 42 --position top3 >> log/run_inet_onlytop3_42.log

CUDA_VISIBLE_DEVICES=3 python main.py --root_path /home/shenxi/Datasets --dataset imagenet --seed 42 --position top6 >> log/run_inet_onlytop6_42.log

CUDA_VISIBLE_DEVICES=4 python main.py --root_path /home/shenxi/Datasets --dataset imagenet --seed 42 --position top9 >> log/run_inet_onlytop9_42.log

CUDA_VISIBLE_DEVICES=4 python main.py --root_path /datassd --dataset imagenet --seed 42 --position top1 >> log/run_inet_onlytop1_42.log

CUDA_VISIBLE_DEVICES=4 python main.py --root_path /datassd --dataset imagenet --seed 42 --position top2 >> log/run_inet_onlytop2_42.log

## Larger R

CUDA_VISIBLE_DEVICES=4 python main.py --root_path /datassd --dataset imagenet --seed 42 --r 4 >> log/run_inet_allr4_42.log

CUDA_VISIBLE_DEVICES=4 python main.py --root_path /datassd --dataset imagenet --seed 42 --r 8 >> log/run_inet_allr8_42.log

## More iters

CUDA_VISIBLE_DEVICES=0 python main.py --root_path /datassd --dataset imagenet --seed 42 --n_iters 750 >> log/run_inet_iters750_42.log

## Remove weight decay

CUDA_VISIBLE_DEVICES=0 python main.py --root_path /datassd --dataset imagenet --seed 42 --weight_decay 0 >> log/run_inet_weightdecay0_42.log
