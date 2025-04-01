# TESS CIFAR10
CUDA_VISIBLE_DEVICES=0 python main.py --dataset CIFAR10 --arch cifar_tessvgg_model --data-path ~/Datasets --save-path ./experiments/CIFAR10_VGG_TESS --trials 1 --epochs 200 --batch-size 128 --val-batch-size 64 --print-freq 20 --delay-ls 6 --factors-stdp 0.2 0.5 1 1 --pooling MAX --scheduler 100 --lr 0.001 --lr-conv 0.001 --experiment-name "CIFAR10" --training-mode tess --loss "CE" --wn --optimizer Adam

# BPTT CIFAR10
CUDA_VISIBLE_DEVICES=0 python main.py --dataset CIFAR10 --arch cifar10_vgg_bptt --data-path ~/Datasets --save-path ./experiments/CIFAR10_VGG_BPTT --trials 1 --epochs 200 --batch-size 128 --val-batch-size 64 --print-freq 20 --pooling MAX --scheduler 100 --lr 0.001 --lr-conv 0.001 --experiment-name "CIFAR10" --training-mode bptt --loss "CE" --wn --optimizer Adam

# TESS CIFAR100
CUDA_VISIBLE_DEVICES=0 python main.py --dataset CIFAR100 --arch cifar100_tessvgg_model --data-path ~/Datasets --save-path ./experiments/CIFAR100_VGG_TESS --trials 1 --epochs 200 --batch-size 128 --val-batch-size 64 --print-freq 20 --delay-ls 6 --factors-stdp 0.2 0.5 1 1 --pooling MAX --scheduler 100 --lr 0.001 --lr-conv 0.001 --experiment-name "CIFAR100" --training-mode tess --loss "CE" --wn --optimizer Adam

# BPTT CIFAR100
CUDA_VISIBLE_DEVICES=0 python main.py --dataset CIFAR100 --arch cifar100_vgg_bptt --data-path ~/Datasets --save-path ./experiments/CIFAR100_VGG_BPTT --trials 1 --epochs 200 --batch-size 128 --val-batch-size 64 --print-freq 20 --pooling MAX --scheduler 100 --lr 0.001 --lr-conv 0.001 --experiment-name "CIFAR100" --training-mode bptt --loss "CE" --wn --optimizer Adam

# TESS DVSCIFAR10
CUDA_VISIBLE_DEVICES=0 python main.py --dataset CIFAR10DVS --arch dvscifar10_tessvgg_model --data-path ~/Datasets --save-path ./experiments/CIFAR10DVS_VGG_TESS --trials 1 --epochs 200 --batch-size 64 --val-batch-size 64 --print-freq 20 --delay-ls 10 --factors-stdp 0.2 0.5 0 1 --pooling MAX --scheduler 100 --lr 0.001 --lr-conv 0.001 --experiment-name "DVSCIFAR10" --training-mode tess --loss "CE" --wn --optimizer Adam

# BPTT DVSCIFAR10
CUDA_VISIBLE_DEVICES=0 python main.py --dataset CIFAR10DVS --arch dvscifar10_vgg_bptt --data-path ~/Datasets --save-path ./experiments/CIFAR10DVS_VGG_BPTT --trials 1 --epochs 200 --batch-size 64 --val-batch-size 64 --print-freq 20 --pooling MAX --scheduler 100 --lr 0.001 --lr-conv 0.001 --experiment-name "DVSCIFAR10" --training-mode bptt --loss "CE" --wn --optimizer Adam

# TESS DVS Gesture
CUDA_VISIBLE_DEVICES=0 python main.py --dataset DVSGesture --arch dvs_tessvgg_model --data-path ~/Datasets --save-path ./experiments/DVSGesture_TESS --trials 1 --epochs 200 --batch-size 16 --val-batch-size 16 --print-freq 20 --delay-ls 20 --factors-stdp 0.2 0.5 0 1 --pooling MAX --scheduler 100 --lr 0.001 --lr-conv 0.001 --experiment-name "DVSGesture" --training-mode tess --loss "CE" --wn --optimizer Adam

# BPTT DVS Gesture
CUDA_VISIBLE_DEVICES=2 python main.py --dataset DVSGesture --arch dvs_vgg_bptt --data-path ~/Datasets --save-path ./experiments/DVSGesture_BPTT --trials 1 --epochs 200 --batch-size 16 --val-batch-size 16 --print-freq 20 --pooling MAX --scheduler 100 --lr 0.001 --lr-conv 0.001 --experiment-name "DVSGesture" --training-mode bptt --loss "CE" --wn --optimizer Adam