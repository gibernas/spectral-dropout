import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from cycler import cycler
file_name = '/home/gianmarco/spectral-dropout/results/evaluation_results.csv'

results_df = pd.read_csv(file_name)
results = results_df.values
print(results[:-1,1:])
# Swap elements so that first column is original dataset and second column is cross dataset
for idx, row in enumerate(results_df.iterrows()):
    if 'sim' in row[1][0]:
        results[idx, 1], results[idx, 2] = results[idx, 2], results[idx, 1]
print(results[:-1,1:])

# We don't care about the simreal dataset here
results = results[:-1, 1:]
x = np.zeros([2, 4]) + 0.25
x[1, :] = 0.75

plt.plot(x, (np.float64(results[:,:].T)), 'x')
plt.legend(['Spectral on Real', 'Vanilla on Real', 'Spectral on Sim', 'Vanilla on Sim'])
plt.xticks([0, 0.25, 0.75, 1], (None, 'Original', 'Cross', '*'))
plt.title('Model evaluations')
plt.xlabel('Dataset')
plt.ylabel('Loss')
plt.savefig('/home/gianmarco/spectral-dropout/results/ugly_plot.png')
plt.show()
print(results)