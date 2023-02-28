
# for RUN_NAME in xent_ls0.05_seed1 xent_ls0.05_seed2 xent_ls0.05_seed3 xent_ls0.05_seed4 xent_ls0.05_seed5 xent_ls0.05_seed
# do
# 	for CORRUPTION_TYPE in impulse_noise shot_noise defocus_blur motion_blur speckle_noise
# 	do
# 		PYTHON_VISIBLE_DEVICES=0 python eval.py --run-name=$RUN_NAME --corruption-type=$CORRUPTION_TYPE --misspecification-cost=4
# 	done
# done


for CORRUPTION_TYPE in shot_noise defocus_blur motion_blur speckle_noise
do
	PYTHON_VISIBLE_DEVICES=0 python eval_ensemble.py --run-name-prefix=rl_mc4 --corruption-type=$CORRUPTION_TYPE --misspecification-cost=4
done