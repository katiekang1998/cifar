import matplotlib.pyplot as plt
import pickle
import numpy as np

corruption_types = ["impulse_noise", "shot_noise", "defocus_blur", "motion_blur", "speckle_noise"]
# rl_run_names = ["rl_mc4_ls0.05_seed1", "rl_mc4_ls0.05_seed2"]
# xent_run_names = ["xent_ls0.05_seed1", "xent_ls0.05_seed2", "xent_ls0.05_seed3", "xent_ls0.05_seed4", "xent_ls0.05_seed5", "xent_ls0.05_seed6"]
rl_run_names = ["rl_mc4_seed"+str(_) for _ in range(1, 7)]
xent_run_names = ["xent_ls0._seed"+str(_) for _ in range(1, 7)]

appdx = "_mc4"

f, ax = plt.subplots(len(corruption_types), 1, figsize=(6, 4*len(corruption_types)))


for corruption_type_idx in range(len(corruption_types)): 
	rewards_all = []
	for rl_run_name in rl_run_names:
		with open("data/"+rl_run_name+"/"+corruption_types[corruption_type_idx]+appdx+'.pkl', 'rb') as f:
		    results = pickle.load(f)
		    rewards_all.append(results["reward"])
	rewards_all = np.array(rewards_all)
	ax[corruption_type_idx].plot([0,1,2,3,4,5], rewards_all.mean(axis=0), c="C0", label = "rl")
	ax[corruption_type_idx].fill_between([0,1,2,3,4,5], rewards_all.mean(axis=0)-rewards_all.std(axis=0), rewards_all.mean(axis=0)+rewards_all.std(axis=0), color="C0", alpha=0.2)

	rewards_all = []
	for xent_run_name in xent_run_names:
		with open("data/"+xent_run_name+"/"+corruption_types[corruption_type_idx]+appdx+'.pkl', 'rb') as f:
		    results = pickle.load(f)
		    rewards_all.append(results["reward"])
	rewards_all = np.array(rewards_all)
	ax[corruption_type_idx].plot([0,1,2,3,4,5], rewards_all.mean(axis=0), c="C1", label = "xent")
	ax[corruption_type_idx].fill_between([0,1,2,3,4,5], rewards_all.mean(axis=0)-rewards_all.std(axis=0), rewards_all.mean(axis=0)+rewards_all.std(axis=0), color="C1", alpha=0.2)

plt.legend()
plt.show()

