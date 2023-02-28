for RUN_NAME in rl_mc4_seed1 rl_mc4_seed2 rl_mc4_seed3 rl_mc4_seed4 rl_mc4_seed5 rl_mc4_seed6
do
	for CORRUPTION_TYPE in impulse_noise shot_noise defocus_blur motion_blur speckle_noise
	do
		PYTHON_VISIBLE_DEVICES=1 python eval.py --run-name=$RUN_NAME --corruption-type=$CORRUPTION_TYPE --misspecification-cost=4
	done
done