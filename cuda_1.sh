for RUN_NAME in xent_ls0._seed1
do
	for CORRUPTION_TYPE in impulse_noise shot_noise defocus_blur motion_blur speckle_noise
	do
		for TEMP in 1.2 1.4 1.6 1.8
		do
			PYTHON_VISIBLE_DEVICES=1 python eval.py --run-name=$RUN_NAME --corruption-type=$CORRUPTION_TYPE --misspecification-cost=4 --temperature=$TEMP
		done
	done
done


# for CORRUPTION_TYPE in impulse_noise shot_noise defocus_blur motion_blur speckle_noise
# do
# 	PYTHON_VISIBLE_DEVICES=1 python eval_ensemble.py --run-name-prefix=xent_ls0. --corruption-type=$CORRUPTION_TYPE --misspecification-cost=4
# done