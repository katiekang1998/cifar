
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

for CORRUPTION_TYPE in impulse_noise shot_noise defocus_blur motion_blur speckle_noise
do
	PYTHON_VISIBLE_DEVICES=1 python eval_ensemble.py --run-name-prefix=xent_ls0. --corruption-type=$CORRUPTION_TYPE --misspecification-cost=4 --dont-use-threshold=True
done

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