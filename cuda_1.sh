# for RUN_NAME in xent_ls0._seed1
# do
# 	for CORRUPTION_TYPE in impulse_noise shot_noise defocus_blur motion_blur speckle_noise
# 	do
# 		for TEMP in 1.2 1.4 1.6 1.8
# 		do
# 			PYTHON_VISIBLE_DEVICES=1 python eval.py --run-name=$RUN_NAME --corruption-type=$CORRUPTION_TYPE --misspecification-cost=4 --temperature=$TEMP
# 		done
# 	done
# done


# for CORRUPTION_TYPE in impulse_noise shot_noise defocus_blur motion_blur speckle_noise
# do
# 	PYTHON_VISIBLE_DEVICES=1 python eval_ensemble.py --run-name-prefix=xent_ls0. --cdorruption-type=$CORRUPTION_TYPE --misspecification-cost=4
# done


# for SEED in 0 1 2 3 4
# do
# 	CUDA_VISIBLE_DEVICES=1 python trainer.py --seed=$SEED --save-dir=xent11_seed$SEED
# done


# for RUN_NAME in xent11_seed0 xent11_seed1 xent11_seed2 xent11_seed3 xent11_seed4
# do
# 	for CORRUPTION_TYPE in impulse_noise shot_noise defocus_blur motion_blur speckle_noise
# 	do
# 		CUDA_VISIBLE_DEVICES=1 python eval_cifar10_outputs_baseline.py --run-name=$RUN_NAME --corruption-type=$CORRUPTION_TYPE
# 	done
# done

# for RUN_NAME in officehome_xent_0 officehome_xent_1 officehome_xent_2 officehome_xent_3 officehome_xent_4
# do
# 	for CORRUPTION_TYPE in Product Clipart Art Photo
# 	do
# 		CUDA_VISIBLE_DEVICES=1 python trainer_ts_oofficehome.py --run-name=$RUN_NAME --val_type=$CORRUPTION_TYPE
# 	done
# done


# for RUN_NAME in officehome_xent66_0 officehome_xent66_1 officehome_xent66_2 officehome_xent66_3 officehome_xent66_4
# do
# 	CUDA_VISIBLE_DEVICES=1 python eval_officehome_outputs_baseline.py --run-name=$RUN_NAME
# done


for RUN_NAME in officehome_xent_0 officehome_xent_1 officehome_xent_2 officehome_xent_3 officehome_xent_4
do
	CUDA_VISIBLE_DEVICES=1 python eval_officehome_outputs_oracle.py --run-name=$RUN_NAME
done