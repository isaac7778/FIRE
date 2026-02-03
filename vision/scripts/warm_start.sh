#!/bin/bash
benchmark="warm_start"

for seed in {1..3}; do
    python train.py --seed "$seed" --benchmark "$benchmark" --model RESNET18 --task CIFAR10
    python train.py --seed "$seed" --benchmark "$benchmark" --model RESNET18 --task CIFAR10 --snp-coef 0.8
    python train.py --seed "$seed" --benchmark "$benchmark" --model RESNET18 --task CIFAR10 --dash-alpha 0.3 --dash-lambda 0.05
    python train.py --seed "$seed" --benchmark "$benchmark" --model RESNET18 --task CIFAR10 --parseval-reg-enable True --parseval-reg-coef 1e-3
    python train.py --seed "$seed" --benchmark "$benchmark" --model RESNET18 --task CIFAR10 --cbp-enable True --cbp-replacement-rate 1e-4 --cbp-maturity-threshold 1000
    python train.py --seed "$seed" --benchmark "$benchmark" --model RESNET18 --task CIFAR10 --redo-enable True --redo-threshold 0.5
    python train.py --seed "$seed" --benchmark "$benchmark" --model RESNET18 --task CIFAR10 --regen-coef 1e-3
    python train.py --seed "$seed" --benchmark "$benchmark" --model RESNET18 --task CIFAR10 --snr-enable True --snr-tau-percentile 0.99
    python train.py --seed "$seed" --benchmark "$benchmark" --model RESNET18 --task CIFAR10 --fire-enable True --fire-iter-num 10
    python train.py --seed "$seed" --benchmark "$benchmark" --model RESNET18 --task CIFAR10 --optimizer muon
    python train.py --seed "$seed" --benchmark "$benchmark" --model RESNET18 --task CIFAR10 --full-reset-enable True

    python train.py --seed "$seed" --benchmark "$benchmark" --model TinyViT --task CIFAR100
    python train.py --seed "$seed" --benchmark "$benchmark" --model TinyViT --task CIFAR100 --snp-coef 0.8
    python train.py --seed "$seed" --benchmark "$benchmark" --model TinyViT --task CIFAR100 --dash-alpha 0.1 --dash-lambda 0.05
    python train.py --seed "$seed" --benchmark "$benchmark" --model TinyViT --task CIFAR100 --parseval-reg-enable True --parseval-reg-coef 1e-5
    python train.py --seed "$seed" --benchmark "$benchmark" --model TinyViT --task CIFAR100 --cbp-enable True --cbp-replacement-rate 1e-4 --cbp-maturity-threshold 100
    python train.py --seed "$seed" --benchmark "$benchmark" --model TinyViT --task CIFAR100 --redo-enable True --redo-threshold 0.5
    python train.py --seed "$seed" --benchmark "$benchmark" --model TinyViT --task CIFAR100 --regen-coef 1e-3
    python train.py --seed "$seed" --benchmark "$benchmark" --model TinyViT --task CIFAR100 --snr-enable True --snr-tau-percentile 0.99
    python train.py --seed "$seed" --benchmark "$benchmark" --model TinyViT --task CIFAR100 --fire-enable True --fire-iter-num 10
    python train.py --seed "$seed" --benchmark "$benchmark" --model TinyViT --task CIFAR100 --optimizer muon
    python train.py --seed "$seed" --benchmark "$benchmark" --model TinyViT --task CIFAR100 --full-reset-enable True

    python train.py --seed "$seed" --benchmark "$benchmark" --model VGG16 --task TinyImageNet
    python train.py --seed "$seed" --benchmark "$benchmark" --model VGG16 --task TinyImageNet --snp-coef 0.8
    python train.py --seed "$seed" --benchmark "$benchmark" --model VGG16 --task TinyImageNet --dash-alpha 0.1 --dash-lambda 0.05
    python train.py --seed "$seed" --benchmark "$benchmark" --model VGG16 --task TinyImageNet --parseval-reg-enable True --parseval-reg-coef 1e-3
    python train.py --seed "$seed" --benchmark "$benchmark" --model VGG16 --task TinyImageNet --cbp-enable True --cbp-replacement-rate 1e-4 --cbp-maturity-threshold 1000
    python train.py --seed "$seed" --benchmark "$benchmark" --model VGG16 --task TinyImageNet --redo-enable True --redo-threshold 0.5
    python train.py --seed "$seed" --benchmark "$benchmark" --model VGG16 --task TinyImageNet --regen-coef 1e-3
    python train.py --seed "$seed" --benchmark "$benchmark" --model VGG16 --task TinyImageNet --snr-enable True --snr-tau-percentile 0.92
    python train.py --seed "$seed" --benchmark "$benchmark" --model VGG16 --task TinyImageNet --fire-enable True --fire-iter-num 10
    python train.py --seed "$seed" --benchmark "$benchmark" --model VGG16 --task TinyImageNet --optimizer muon
    python train.py --seed "$seed" --benchmark "$benchmark" --model VGG16 --task TinyImageNet --full-reset-enable True
done