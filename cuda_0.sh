
for RUN_NAME in rl_mc8
do
	for CORRUPTION_TYPE in impulse_noise shot_noise defocus_blur motion_blur speckle_noise
	do
		PYTHON_VISIBLE_DEVICES=0 python eval.py --run-name=$RUN_NAME --corruption-type=$CORRUPTION_TYPE --misspecification-cost=4
	done
done