import matplotlib.pyplot as plt
import pickle

corruption_types = ["impulse_noise", "shot_noise", "defocus_blur", "motion_blur", "speckle_noise"]
appdx = "_mc4"

f, ax = plt.subplots(len(corruption_types), 1, figsize=(1*6, 4*len(corruption_types)))
# for run_name in ["rl_mc4_ls0.05_seed1", "rl_mc4_ls0.05_seed2", #"rl_mc4_seed3", "rl_mc4_seed4", "rl_mc4_seed5", "rl_mc4_seed6", 
# "xent_ls0.05_seed1", "xent_ls0.05_seed2", "xent_ls0.05_seed3", "xent_ls0.05_seed4", "xent_ls0.05_seed5", "xent_ls0.05_seed6"]:

for run_name in ["rl_mc4_ensemble_eval"]:
	for corruption_type_idx in range(len(corruption_types)): 
		with open("data/"+run_name+"/"+corruption_types[corruption_type_idx]+appdx+'.pkl', 'rb') as f:
			results = pickle.load(f)


		ax[corruption_type_idx].plot([0, 1, 2, 3, 4, 5], results["reward"], label = run_name)
		ax[corruption_type_idx].set_title(corruption_types[corruption_type_idx])
		ax[corruption_type_idx].set_ylabel("reward")
		ax[corruption_type_idx].legend()


for run_name in ["xent_ls0._ensemble_eval"]:
	for corruption_type_idx in range(len(corruption_types)): 
		with open("data/"+run_name+"/"+corruption_types[corruption_type_idx]+appdx+'_ts.pkl', 'rb') as f:
			results = pickle.load(f)
			best_results = results
		# 	best_t = 1
		# for t in [1.1, 1.2, 1.3, 1.4, 1.5,]:
		# 	with open("data/"+run_name+"/"+corruption_types[corruption_type_idx]+appdx+"_ts"+str(t)+'.pkl', 'rb') as f:
		# 		results = pickle.load(f)
		# 		if len(best_results.keys())==0 or results["reward"][0]>best_results["reward"][0]:
		# 			best_results = results
		# 			best_t = t
		# print(best_t)

		ax[corruption_type_idx].plot([0, 1, 2, 3, 4, 5], best_results["reward"], label = run_name+"_ts")

		with open("data/"+run_name+"/"+corruption_types[corruption_type_idx]+appdx+'_oracle_threshold.pkl', 'rb') as f:
			results = pickle.load(f)
			best_results = results

		ax[corruption_type_idx].plot([0, 1, 2, 3, 4, 5], best_results["reward"], label = run_name+"_oracle_threshold")

		with open("data/"+run_name+"/"+corruption_types[corruption_type_idx]+appdx+'.pkl', 'rb') as f:
			results = pickle.load(f)
			best_results = results

		ax[corruption_type_idx].plot([0, 1, 2, 3, 4, 5], best_results["reward"], label = run_name)

		with open("data/"+run_name+"/"+corruption_types[corruption_type_idx]+appdx+'_no_threshold.pkl', 'rb') as f:
			results = pickle.load(f)
			best_results = results

		ax[corruption_type_idx].plot([0, 1, 2, 3, 4, 5], best_results["reward"], label = run_name+"_no_threshold")

		ax[corruption_type_idx].set_title(corruption_types[corruption_type_idx])
		ax[corruption_type_idx].set_ylabel("reward")
		ax[corruption_type_idx].legend()

# for run_name in ["rl_mc4_seed1"]:
# 	for corruption_type_idx in range(len(corruption_types)): 
# 		with open("data/"+run_name+"/"+corruption_types[corruption_type_idx]+appdx+'.pkl', 'rb') as f:
# 		    results = pickle.load(f)

# 		if "rl" in run_name:
# 			c = "C0"
# 		else:
# 			c = "C1"
# 		ax[corruption_type_idx].plot([0, 1, 2, 3, 4, 5], results["reward"], label = run_name, c=c)
# 		ax[corruption_type_idx].set_title(corruption_types[corruption_type_idx])
# 		ax[corruption_type_idx].set_ylabel("reward")
# 		ax[corruption_type_idx].legend()

# plt.show()
plt.savefig("data/plot.png")