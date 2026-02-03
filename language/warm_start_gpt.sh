num_gpu=8

for seed in {1..3}; do
    # warm-started model
    torchrun --standalone --nproc_per_node="$num_gpu" train.py --seed="$seed" --method_type=vanilla \
    --c0_dataset=wikitext --c0_subset_ratio=1.0 --c0_data_replay_ratio=400 \
    --c1_dataset=wiki_owt --c1_subset_ratio=0.0 --c1_data_replay_ratio=0 \
    
    # full reset model
    torchrun --standalone --nproc_per_node="$num_gpu" train_warm_start.py --seed="$seed" --method_type=full_reset

    # best ckpt
    torchrun --standalone --nproc_per_node="$num_gpu" train.py --seed="$seed" --method_type=vanilla \
    --warm_start_load_path=output/vanilla_seed"$seed"_wikitext_1.0_400_wiki_owt_0.0_0/best_chunk0_ckpt.pt \
    --comment=_best;
    torchrun --standalone --nproc_per_node="$num_gpu" train.py --seed="$seed" --method_type=snp \
    --warm_start_load_path=output/vanilla_seed"$seed"_wikitext_1.0_400_wiki_owt_0.0_0/best_chunk0_ckpt.pt \
    --snp_init_load_path=output/vanilla_seed"$seed"_wikitext_1.0_400_wiki_owt_0.0_0/init_ckpt.pt \
    --snp_shrink_coef=0.5 \
    --comment=_best;
    torchrun --standalone --nproc_per_node="$num_gpu" train.py --seed="$seed" --method_type=fire \
    --warm_start_load_path=output/vanilla_seed"$seed"_wikitext_1.0_400_wiki_owt_0.0_0/best_chunk0_ckpt.pt \
    --fire_iteration=5 \
    --comment=_best;

    # 30k iter ckpt
    torchrun --standalone --nproc_per_node="$num_gpu" train.py --seed="$seed" --method_type=vanilla \
    --warm_start_load_path=output/vanilla_seed"$seed"_wikitext_1.0_400_wiki_owt_0.0_0/chunk0_iter30000_ckpt.pt \
    --comment=_30k;
    torchrun --standalone --nproc_per_node="$num_gpu" train.py --seed="$seed" --method_type=snp \
    --warm_start_load_path=output/vanilla_seed"$seed"_wikitext_1.0_400_wiki_owt_0.0_0/chunk0_iter30000_ckpt.pt \
    --snp_init_load_path=output/vanilla_seed"$seed"_wikitext_1.0_400_wiki_owt_0.0_0/init_ckpt.pt \
    --snp_shrink_coef=0.8 \
    --comment=_30k;
    torchrun --standalone --nproc_per_node="$num_gpu" train.py --seed="$seed" --method_type=fire \
    --warm_start_load_path=output/vanilla_seed"$seed"_wikitext_1.0_400_wiki_owt_0.0_0/chunk0_iter30000_ckpt.pt \
    --fire_iteration=5 \
    --comment=_30k;

    # 60k iter ckpt
    torchrun --standalone --nproc_per_node="$num_gpu" train.py --seed="$seed" --method_type=vanilla \
    --warm_start_load_path=output/vanilla_seed"$seed"_wikitext_1.0_400_wiki_owt_0.0_0/chunk0_iter60000_ckpt.pt \
    --comment=_60k;
    torchrun --standalone --nproc_per_node="$num_gpu" train.py --seed="$seed" --method_type=snp \
    --warm_start_load_path=output/vanilla_seed"$seed"_wikitext_1.0_400_wiki_owt_0.0_0/chunk0_iter60000_ckpt.pt \
    --snp_init_load_path=output/vanilla_seed"$seed"_wikitext_1.0_400_wiki_owt_0.0_0/init_ckpt.pt \
    --snp_shrink_coef=0.5 \
    --comment=_60k;
    torchrun --standalone --nproc_per_node="$num_gpu" train.py --seed="$seed" --method_type=fire \
    --warm_start_load_path=output/vanilla_seed"$seed"_wikitext_1.0_400_wiki_owt_0.0_0/chunk0_iter60000_ckpt.pt \
    --fire_iteration=5 \
    --comment=_60k;