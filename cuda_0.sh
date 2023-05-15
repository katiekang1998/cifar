
# for RUN_NAME in xent_ls0.05_seed1 xent_ls0.05_seed2 xent_ls0.05_seed3 xent_ls0.05_seed4 xent_ls0.05_seed5 xent_ls0.05_seed
# do
# 	for CORRUPTION_TYPE in impulse_noise shot_noise defocus_blur motion_blur speckle_noise
# 	do
# 		PYTHON_VISIBLE_DEVICES=0 python eval.py --run-name=$RUN_NAME --corruption-type=$CORRUPTION_TYPE --misspecification-cost=4
# 	done
# done

# for MC in 1 8 12
# do
# 	for CORRUPTION_TYPE in impulse_noise shot_noise defocus_blur motion_blur speckle_noise
# 	do
# 		for SEED in 1 2 3 4 5 6
# 		do
# 			PYTHON_VISIBLE_DEVICES=0 python eval.py --run-name=rl_mc${MC}_seed${SEED} --corruption-type=$CORRUPTION_TYPE --misspecification-cost=$MC
# 			PYTHON_VISIBLE_DEVICES=2 python eval.py --run-name=xent_ls0._seed${SEED} --corruption-type=$CORRUPTION_TYPE --misspecification-cost=$MC
# 		done
# 	done
# done

# for RUN_NAME in rl_mc4_seed1 rl_mc4_seed2 rl_mc4_seed3 rl_mc4_seed4 rl_mc4_seed5
# do
# 	for CORRUPTION_TYPE in impulse_noise shot_noise defocus_blur motion_blur speckle_noise
# 	do
# 		PYTHON_VISIBLE_DEVICES=1 python eval_cifar10_outputs.py --run-name=$RUN_NAME --corruption-type=$CORRUPTION_TYPE
# 	done
# done

# for RUN_NAME in poverty2_49 poverty2_51 poverty2_52 poverty2_53 poverty2_54
# do
# 	CUDA_VISIBLE_DEVICES=0 python eval_poverty.py --run-name=$RUN_NAME
# done


# for SEED in 49 50 51 52 53
# do
# 	CUDA_VISIBLE_DEVICES=0 python trainer_wilds.py --seed=$SEED --save-dir=poverty2_$SEED
# done

# for CORRUPTION_TYPE in impulse_noise shot_noise defocus_blur motion_blur speckle_noise
# do
# 	PYTHON_VISIBLE_DEVICES=2 python eval_ensemble.py --run-name-prefix=rl_mc12 --corruption-type=$CORRUPTION_TYPE --misspecification-cost=12
# 	PYTHON_VISIBLE_DEVICES=2 python eval_ensemble.py --run-name-prefix=xent_ls0.01 --corruption-type=$CORRUPTION_TYPE --misspecification-cost=12 --ts=True
# done


# for CORRUPTION_TYPE in impulse_noise shot_noise defocus_blur motion_blur speckle_noise
# do
# 	for CORRUPTION_LEVEL in 0 1 2 3 4
# 	do
# 		PYTHON_VISIBLE_DEVICES=0 python trainer_ts.py --run-name-prefix=xent_ls0. --corruption-type=$CORRUPTION_TYPE --corruption-level=$CORRUPTION_LEVEL
# 	done
# done


# PYTHON_VISIBLE_DEVICES=0 python trainer_ts.py --run-name-prefix=xent_ls0.


for RUN_NAME in xent_ls0._seed1 xent_ls0._seed2 xent_ls0._seed3 xent_ls0._seed4 xent_ls0._seed5
do
	for CORRUPTION_TYPE in impulse_noise shot_noise defocus_blur motion_blur speckle_noise
	do
		python eval_cifar10_outputs_oracle.py --run-name=$RUN_NAME --corruption-type=$CORRUPTION_TYPE
	done
done


# for RUN_NAME in xent_ls0._seed1 xent_ls0._seed2 xent_ls0._seed3 xent_ls0._seed4 xent_ls0._seed5
# do
# 	python trainer_ts.py --run-name=$RUN_NAME
# 	for CORRUPTION_TYPE in impulse_noise shot_noise defocus_blur motion_blur speckle_noise
# 	do
# 		for CORRUPTION_LEVEL in 0 1 2 3 4
# 		do
# 			python trainer_ts.py --run-name=$RUN_NAME --corruption-type=$CORRUPTION_TYPE --corruption-level=$CORRUPTION_LEVEL
# 		done
# 	done
# done