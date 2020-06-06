import matplotlib.pyplot as plt
from stable_baselines import results_plotter

from config import TIME_STEPS

log_dir = "./monitor_logs/"
results_plotter.plot_results([log_dir], TIME_STEPS, results_plotter.X_TIMESTEPS, "Rewards over episodes")
plt.show()