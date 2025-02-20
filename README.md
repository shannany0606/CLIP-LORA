# CLIP-LORA

CUDA_VISIBLE_DEVICES=5 python main.py --root_path /home/shenxi/Datasets --dataset stanford_cars --seed 42 --params k v >> log/run_car_rm_q_42.log

CUDA_VISIBLE_DEVICES=5 python main.py --root_path /home/shenxi/Datasets --dataset stanford_cars --seed 42 --params q v >> log/run_car_rm_k_42.log

CUDA_VISIBLE_DEVICES=8 python main.py --root_path /home/shenxi/Datasets --dataset stanford_cars --seed 42 --params q k >> log/run_car_rm_v_42.log

CUDA_VISIBLE_DEVICES=5 python main.py --root_path /home/shenxi/Datasets --dataset food101 --seed 42 >> log/run_food_ema42.log

CUDA_VISIBLE_DEVICES=4 python main.py --root_path /home/shenxi/Datasets --dataset imagenet --seed 42 >> log/run_Inet_42.log

CUDA_VISIBLE_DEVICES=1 python main.py --root_path /home/shenxi/Datasets --dataset imagenet --seed 42 --params k v >> log/run_Inet_rm_q_42.log

CUDA_VISIBLE_DEVICES=2 python main.py --root_path /home/shenxi/Datasets --dataset imagenet --seed 42 --params q v >> log/run_Inet_rm_k_42.log

CUDA_VISIBLE_DEVICES=3 python main.py --root_path /home/shenxi/Datasets --dataset imagenet --seed 42 --params q k >> log/run_Inet_rm_v_42.log