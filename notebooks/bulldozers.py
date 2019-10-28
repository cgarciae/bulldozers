#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


#%%

df = pd.read_csv("inputs/bluebook-for-bulldozers/TrainAndValid.csv")

df["SalePrice"] = np.log(df["SalePrice"])
df[df.YearMade < 1800]

# %%
df.describe()


# %%
print(len())
plt.hist(df[df.YearMade > 1800].YearMade)
