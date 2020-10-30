import math
import glob
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import urllib3
from sklearn.externals import joblib
import random
import matplotlib
import sys
sys.path.append(r"C:\DevSource\Shuyang-GEOG676\Projects\sompy")
from sompy import SOMFactory
from plot_tools import plot_hex_map
import logging

df = pd.read_csv(r"C:\Users\Allen\Desktop\Crash/1.csv")

df = df[["CrashMonth","CrashDay", "CrashTime"]]
clustering_vars = ["CrashMonth","CrashDay", "CrashTime"]
df = df.fillna(0)
data = df[clustering_vars].values
names = clustering_vars
df.describe()

for i in range(100):
    sm = SOMFactory().build(data, mapsize=[random.choice(list(range(15, 25))), 
                                           random.choice(list(range(10, 15)))],
                            normalization = 'var', initialization='random', component_names=names, lattice="hexa")
    sm.train(n_job=4, verbose=False, train_rough_len=30, train_finetune_len=100)
    joblib.dump(sm, "model_{}.joblib".format(i))
    
models_pool = glob.glob("./model*")
errors=[]
for model_filepath in models_pool:
    sm = joblib.load(model_filepath)
    topographic_error = sm.calculate_topographic_error()
    quantization_error = sm.calculate_quantization_error()
    errors.append((topographic_error, quantization_error))
e_top, e_q = zip(*errors)

plt.scatter(e_top, e_q)
plt.xlabel("Topographic error")
plt.ylabel("Quantization error")
plt.show()

selected_model = 3
sm = joblib.load(models_pool[selected_model])

topographic_error = sm.calculate_topographic_error()
quantization_error = sm.calculate_quantization_error()
print ("Topographic error = %s\n Quantization error = %s" % (topographic_error, quantization_error))

from mapview import View2D
view2D  = View2D(10,10,"", text_size=7)
view2D.show(sm, col_sz=5, which_dim="all", denormalize=True)
plt.show()

from bmuhits import BmuHitsView
vhts  = BmuHitsView(12,12,"Hits Map",text_size=7)
vhts.show(sm, anotate=True, onlyzeros=False, labelsize=7, cmap="autumn", logaritmic=False)
plt.show()