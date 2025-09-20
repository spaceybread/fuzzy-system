from collections import defaultdict
import matplotlib.pyplot as plt
from sklearn import metrics
import pandas as pd
import numpy as np
import random
import sys

rad_df = pd.read_csv(sys.argv[1])
gaus_df = pd.read_csv(sys.argv[2])

gaus_df = gaus_df.sort_values(by='TMR')
rad_df = rad_df.sort_values(by='TMR')

tmr_r = rad_df['TMR'].to_numpy()
fmr_r = np.log(rad_df['FMR'].to_numpy()) / np.log(2)

tmr_g = gaus_df['TMR'].to_numpy()
fmr_g = np.log(gaus_df['FMR'].to_numpy()) / np.log(2)

plt.figure(figsize=(8, 6))
plt.plot(fmr_r, tmr_r, color='darkorange', lw=2)
plt.plot(fmr_g, tmr_g, color='blue', lw=2, linestyle='--')
plt.ylim([0.0, 1.0])
plt.xlabel('FMR')
plt.ylabel('TMR')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()

