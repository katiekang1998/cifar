import matplotlib.pyplot as plt
import pickle

corruption_types = ["impulse_noise", "shot_noise", "defocus_blur", "motion_blur", "speckle_noise"]
appdx = "_mc4"

f, ax = plt.subplots(len(corruption_types), 3, figsize=(3*6, 4*len(corruption_types)))
for run_name in ["rl_mc4_ls0.05_seed1", "rl_mc4_ls0.05_seed2", #"rl_mc4_seed3", "rl_mc4_seed4", "rl_mc4_seed5", "rl_mc4_seed6", 
"xent_ls0.05_seed1", "xent_ls0.05_seed2", "xent_ls0.05_seed3", "xent_ls0.05_seed4", "xent_ls0.05_seed5", "xent_ls0.05_seed6"]:

# for run_name in ["rl_mc4_ensemble_eval", "xent_ls0._ensemble_eval"]:
	for corruption_type_idx in range(len(corruption_types)): 
		with open("data/"+run_name+"/"+corruption_types[corruption_type_idx]+appdx+'.pkl', 'rb') as f:
		    results = pickle.load(f)

		if "rl" in run_name:
			c = "C0"
		else:
			c = "C1"
		ax[corruption_type_idx][0].plot([0, 1, 2, 3, 4, 5], results["reward"], label = run_name, c=c)
		ax[corruption_type_idx][0].set_title(corruption_types[corruption_type_idx])
		ax[corruption_type_idx][0].set_ylabel("reward")
		# ax[corruption_type_idx][0].legend()
		ax[corruption_type_idx][1].plot([0, 1, 2, 3, 4, 5], results["accuracy"], label = run_name)
		ax[corruption_type_idx][1].set_ylabel("accuracy")
		ax[corruption_type_idx][2].plot([0, 1, 2, 3, 4, 5], results["a10_ratio"], label = run_name)
		ax[corruption_type_idx][2].set_ylabel("a10_ratio")

plt.show()