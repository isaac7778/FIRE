run() {
  echo
  echo
  echo "$@"
  "$@" || exit 1
}

# run python train.py --benchmark warm_start --model RESNET18 --task CIFAR10 --disable-wandb True
# run python train.py --benchmark warm_start --model RESNET18 --task CIFAR10 --disable-wandb True --optimizer muon
# run python train.py --benchmark warm_start --model RESNET18 --task CIFAR10 --disable-wandb True --regen-coef 1
# run python train.py --benchmark warm_start --model RESNET18 --task CIFAR10 --disable-wandb True --snp-coef 1
# run python train.py --benchmark warm_start --model RESNET18 --task CIFAR10 --disable-wandb True --dash-alpha 1 --dash-lambda 1
# run python train.py --benchmark warm_start --model RESNET18 --task CIFAR10 --disable-wandb True --full-reset-enable True
# run python train.py --benchmark warm_start --model RESNET18 --task CIFAR10 --disable-wandb True --parseval-reg-enable True
# run python train.py --benchmark warm_start --model RESNET18 --task CIFAR10 --disable-wandb True --fire-enable True
# run python train.py --benchmark warm_start --model RESNET18 --task CIFAR10 --disable-wandb True --cbp-enable True
# run python train.py --benchmark warm_start --model RESNET18 --task CIFAR10 --disable-wandb True --redo-enable True
# run python train.py --benchmark warm_start --model RESNET18 --task CIFAR10 --disable-wandb True --snr-enable True
# run python train.py --benchmark warm_start --model TinyViT  --task CIFAR100 --disable-wandb True
# run python train.py --benchmark warm_start --model TinyViT  --task CIFAR100 --disable-wandb True --optimizer muon
# run python train.py --benchmark warm_start --model TinyViT  --task CIFAR100 --disable-wandb True --regen-coef 1
# run python train.py --benchmark warm_start --model TinyViT  --task CIFAR100 --disable-wandb True --snp-coef 1
# run python train.py --benchmark warm_start --model TinyViT  --task CIFAR100 --disable-wandb True --dash-alpha 1 --dash-lambda 1
# run python train.py --benchmark warm_start --model TinyViT  --task CIFAR100 --disable-wandb True --full-reset-enable True
# run python train.py --benchmark warm_start --model TinyViT  --task CIFAR100 --disable-wandb True --parseval-reg-enable True
# run python train.py --benchmark warm_start --model TinyViT  --task CIFAR100 --disable-wandb True --fire-enable True
# run python train.py --benchmark warm_start --model TinyViT  --task CIFAR100 --disable-wandb True --cbp-enable True
# run python train.py --benchmark warm_start --model TinyViT  --task CIFAR100 --disable-wandb True --redo-enable True
# run python train.py --benchmark warm_start --model TinyViT  --task CIFAR100 --disable-wandb True --snr-enable True
# run python train.py --benchmark warm_start --model VGG16  --task TinyImageNet --disable-wandb True
# run python train.py --benchmark warm_start --model VGG16  --task TinyImageNet --disable-wandb True --optimizer muon
# run python train.py --benchmark warm_start --model VGG16  --task TinyImageNet --disable-wandb True --regen-coef 1
# run python train.py --benchmark warm_start --model VGG16  --task TinyImageNet --disable-wandb True --snp-coef 1
# run python train.py --benchmark warm_start --model VGG16  --task TinyImageNet --disable-wandb True --dash-alpha 1 --dash-lambda 1
# run python train.py --benchmark warm_start --model VGG16  --task TinyImageNet --disable-wandb True --full-reset-enable True
# run python train.py --benchmark warm_start --model VGG16  --task TinyImageNet --disable-wandb True --parseval-reg-enable True
# run python train.py --benchmark warm_start --model VGG16  --task TinyImageNet --disable-wandb True --fire-enable True
# run python train.py --benchmark warm_start --model VGG16  --task TinyImageNet --disable-wandb True --cbp-enable True
# run python train.py --benchmark warm_start --model VGG16  --task TinyImageNet --disable-wandb True --redo-enable True
# run python train.py --benchmark warm_start --model VGG16  --task TinyImageNet --disable-wandb True --snr-enable True

# run python train.py --benchmark continual --model RESNET18 --task CIFAR10 --disable-wandb True
# run python train.py --benchmark continual --model RESNET18 --task CIFAR10 --disable-wandb True --optimizer muon
# run python train.py --benchmark continual --model RESNET18 --task CIFAR10 --disable-wandb True --regen-coef 1
# run python train.py --benchmark continual --model RESNET18 --task CIFAR10 --disable-wandb True --snp-coef 1
# run python train.py --benchmark continual --model RESNET18 --task CIFAR10 --disable-wandb True --dash-alpha 1 --dash-lambda 1
# run python train.py --benchmark continual --model RESNET18 --task CIFAR10 --disable-wandb True --full-reset-enable True
# run python train.py --benchmark continual --model RESNET18 --task CIFAR10 --disable-wandb True --parseval-reg-enable True
# run python train.py --benchmark continual --model RESNET18 --task CIFAR10 --disable-wandb True --fire-enable True
# run python train.py --benchmark continual --model RESNET18 --task CIFAR10 --disable-wandb True --cbp-enable True
# run python train.py --benchmark continual --model RESNET18 --task CIFAR10 --disable-wandb True --redo-enable True
# run python train.py --benchmark continual --model RESNET18 --task CIFAR10 --disable-wandb True --snr-enable True
# run python train.py --benchmark continual --model TinyViT  --task CIFAR100 --disable-wandb True
# run python train.py --benchmark continual --model TinyViT  --task CIFAR100 --disable-wandb True --optimizer muon
# run python train.py --benchmark continual --model TinyViT  --task CIFAR100 --disable-wandb True --regen-coef 1
# run python train.py --benchmark continual --model TinyViT  --task CIFAR100 --disable-wandb True --snp-coef 1
# run python train.py --benchmark continual --model TinyViT  --task CIFAR100 --disable-wandb True --dash-alpha 1 --dash-lambda 1
# run python train.py --benchmark continual --model TinyViT  --task CIFAR100 --disable-wandb True --full-reset-enable True
# run python train.py --benchmark continual --model TinyViT  --task CIFAR100 --disable-wandb True --parseval-reg-enable True
# run python train.py --benchmark continual --model TinyViT  --task CIFAR100 --disable-wandb True --fire-enable True
# run python train.py --benchmark continual --model TinyViT  --task CIFAR100 --disable-wandb True --cbp-enable True
# run python train.py --benchmark continual --model TinyViT  --task CIFAR100 --disable-wandb True --redo-enable True
# run python train.py --benchmark continual --model TinyViT  --task CIFAR100 --disable-wandb True --snr-enable True
# run python train.py --benchmark continual --model VGG16  --task TinyImageNet --disable-wandb True
# run python train.py --benchmark continual --model VGG16  --task TinyImageNet --disable-wandb True --optimizer muon
# run python train.py --benchmark continual --model VGG16  --task TinyImageNet --disable-wandb True --regen-coef 1
# run python train.py --benchmark continual --model VGG16  --task TinyImageNet --disable-wandb True --snp-coef 1
# run python train.py --benchmark continual --model VGG16  --task TinyImageNet --disable-wandb True --dash-alpha 1 --dash-lambda 1
# run python train.py --benchmark continual --model VGG16  --task TinyImageNet --disable-wandb True --full-reset-enable True
# run python train.py --benchmark continual --model VGG16  --task TinyImageNet --disable-wandb True --parseval-reg-enable True
# run python train.py --benchmark continual --model VGG16  --task TinyImageNet --disable-wandb True --fire-enable True
# run python train.py --benchmark continual --model VGG16  --task TinyImageNet --disable-wandb True --cbp-enable True
# run python train.py --benchmark continual --model VGG16  --task TinyImageNet --disable-wandb True --redo-enable True
# run python train.py --benchmark continual --model VGG16  --task TinyImageNet --disable-wandb True --snr-enable True

# run python train.py --benchmark class_incremental --model TinyViT  --task CIFAR100 --disable-wandb True
# run python train.py --benchmark class_incremental --model TinyViT  --task CIFAR100 --disable-wandb True --optimizer muon
# run python train.py --benchmark class_incremental --model TinyViT  --task CIFAR100 --disable-wandb True --regen-coef 1
# run python train.py --benchmark class_incremental --model TinyViT  --task CIFAR100 --disable-wandb True --snp-coef 1
# run python train.py --benchmark class_incremental --model TinyViT  --task CIFAR100 --disable-wandb True --dash-alpha 1 --dash-lambda 1
# run python train.py --benchmark class_incremental --model TinyViT  --task CIFAR100 --disable-wandb True --full-reset-enable True
# run python train.py --benchmark class_incremental --model TinyViT  --task CIFAR100 --disable-wandb True --parseval-reg-enable True
# run python train.py --benchmark class_incremental --model TinyViT  --task CIFAR100 --disable-wandb True --fire-enable True
# run python train.py --benchmark class_incremental --model TinyViT  --task CIFAR100 --disable-wandb True --cbp-enable True
# run python train.py --benchmark class_incremental --model TinyViT  --task CIFAR100 --disable-wandb True --redo-enable True
# run python train.py --benchmark class_incremental --model TinyViT  --task CIFAR100 --disable-wandb True --snr-enable True
# run python train.py --benchmark class_incremental --model VGG16  --task TinyImageNet --disable-wandb True
# run python train.py --benchmark class_incremental --model VGG16  --task TinyImageNet --disable-wandb True --optimizer muon
# run python train.py --benchmark class_incremental --model VGG16  --task TinyImageNet --disable-wandb True --regen-coef 1
# run python train.py --benchmark class_incremental --model VGG16  --task TinyImageNet --disable-wandb True --snp-coef 1
# run python train.py --benchmark class_incremental --model VGG16  --task TinyImageNet --disable-wandb True --dash-alpha 1 --dash-lambda 1
# run python train.py --benchmark class_incremental --model VGG16  --task TinyImageNet --disable-wandb True --full-reset-enable True
# run python train.py --benchmark class_incremental --model VGG16  --task TinyImageNet --disable-wandb True --parseval-reg-enable True
# run python train.py --benchmark class_incremental --model VGG16  --task TinyImageNet --disable-wandb True --fire-enable True
# run python train.py --benchmark class_incremental --model VGG16  --task TinyImageNet --disable-wandb True --cbp-enable True
# run python train.py --benchmark class_incremental --model VGG16  --task TinyImageNet --disable-wandb True --redo-enable True
# run python train.py --benchmark class_incremental --model VGG16  --task TinyImageNet --disable-wandb True --snr-enable True

