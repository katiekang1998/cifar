import matplotlib.pyplot as plt
import pickle

corruption_types = ["impulse_noise", "shot_noise", "defocus_blur", "motion_blur", "speckle_noise"]
appdx = "_mc2"

f, ax = plt.subplots(len(corruption_types), 3, figsize=(3*6, 4*len(corruption_types)))
for run_name in ["rl_mc2", "xent", "xent_ls0pt1", "xent_ls0pt05", "xent_ls0pt03", "xent_ls0pt01", "rl_mc4"]:
	for corruption_type_idx in range(len(corruption_types)): 
		with open(run_name+"/"+corruption_types[corruption_type_idx]+appdx+'.pkl', 'rb') as f:
		    results = pickle.load(f)
		ax[corruption_type_idx][0].plot([0, 1, 2, 3, 4, 5], results["reward"], label = run_name)
		ax[corruption_type_idx][0].set_title(corruption_types[corruption_type_idx])
		ax[corruption_type_idx][0].set_ylabel("reward")
		ax[corruption_type_idx][0].legend()
		ax[corruption_type_idx][1].plot([0, 1, 2, 3, 4, 5], results["accuracy"], label = run_name)
		ax[corruption_type_idx][1].set_ylabel("accuracy")
		ax[corruption_type_idx][2].plot([0, 1, 2, 3, 4, 5], results["a10_ratio"], label = run_name)
		ax[corruption_type_idx][2].set_ylabel("a10_ratio")

plt.show()