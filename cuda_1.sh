for RUN_NAME in xent_ls0._seed1 xent_ls0._seed2 xent_ls0._seed3 xent_ls0._seed4 xent_ls0._seed5 xent_ls0._seed6
do
	for CORRUPTION_TYPE in impulse_noise shot_noise defocus_blur motion_blur speckle_noise
	do
		PYTHON_VISIBLE_DEVICES=1 python eval.py --run-name=$RUN_NAME --corruption-type=$CORRUPTION_TYPE --misspecification-cost=4
	done
done


# for CORRUPTION_TYPE in impulse_noise shot_noise defocus_blur motion_blur speckle_noise
# do
# 	PYTHON_VISIBLE_DEVICES=1 python eval_ensemble.py --run-name-prefix=xent_ls0. --corruption-type=$CORRUPTION_TYPE --misspecification-cost=4
# done