import numpy as np
import pickle
import matplotlib.pyplot as plt

plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams['font.family'] = 'STIXGeneral'
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False
colors = ['coral','indigo','springgreen']

def main():
    # Load the data
    with open('logs/new_data.pkl', 'rb') as f:
        data = pickle.load(f)
    
    Ns = data['Ns']
    N_samples = data['N_samples']
    modes = data['modes']
    backends = data['backends']
    N_iterations = data['N_iterations']
    N_walkers = data['N_walkers']   



    ncols = len(modes)
    fig, axes = plt.subplots(1, ncols, figsize=(3*ncols,2), sharex=True,layout='constrained')

    for i, mode in enumerate(modes):
        ax = axes[i]
        for ib , backend in enumerate(backends):
            col = colors[ib]
            times = data[mode][backend]
            mean_times = np.mean(times, axis=0)
            std_times = np.std(times, axis=0)
            ax.errorbar(Ns, mean_times, yerr=std_times, label=backend, marker='.', capsize=2,color=col)
        ax.set_xscale('log',base=2)
        ax.set_yscale('log')
        ax.set_xlabel('Number of spins $N$')
        if i == 0:
            ax.set_ylabel('Time (s)')
        ax.set_title(f'Mode: {mode}')
        ax.set_xlim(0.9*min(Ns), 1.1*max(Ns))
        # ax.legend()
    
    axes[0].legend(frameon=False,fontsize=8)
    title = rf"Time performance for $N_{{samples}} = {N_samples}$ and averaged over {N_iterations} iterations. $N_{{walkers}} = {N_walkers}$ for 'multi' modes"
    fig.suptitle(title)
    fig.savefig('logs/new_timing.png', dpi=300, bbox_inches='tight')
    print(data.keys())


if __name__ == "__main__":
    main()