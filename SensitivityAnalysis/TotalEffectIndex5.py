import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from SALib.sample import saltelli
from SALib.analyze import delta

def model(data):
    coef_for_environment = 0.1
    coef_for_others = (1 - coef_for_environment) / (5 - 1)

    y = (data[:, 1] * coef_for_environment + coef_for_others *
       (data[:, 0] + data[:, 2] + data[:, 3] + data[:, 4]))

    return y

year = 2015
data = pd.read_excel("D:/КПІ/3-курс/Data Science/project/DS_IndexProsperity/SensitivityAnalysis_Result/" + f"{year}.xlsx")
Indexs5 = data.iloc[:, 20:25]

minValue = Indexs5.min()
maxValue = Indexs5.max()

problem = {
    'num_vars': 5,  #Кількість параметрів
    'names': ['Economic',  'Environment', 'Education', 'Health', 'Safety'],  #Назви параметрів
    'bounds': [(minValue.iloc[0], maxValue.iloc[0]),
               (minValue.iloc[1], maxValue.iloc[1]),
               (minValue.iloc[2], maxValue.iloc[2]),
               (minValue.iloc[3], maxValue.iloc[3]),
               (minValue.iloc[4], maxValue.iloc[4])], #Межі для кожного параметра
}

param = Indexs5.sample(n = 5000, replace = True, random_state=2).values

TAI = delta.analyze(problem, param, model(param))
print('Total-Effect Indices:', TAI['delta'])

namePlot = "Total-Effect indices for 5 indexs"
df_result = pd.DataFrame({
    "variable" : list(Indexs5.columns),
    "Total-Effect Indices" : TAI['delta'],
})

ax = df_result.plot(x="indices", kind="bar", stacked=True, rot=0, title="Total-Effect Indices")
plt.savefig('D:/КПІ/3-курс/Data Science/project/DS_IndexProsperity/SensitivityAnalysis_Result/' + f'{namePlot}.png')
plt.show()

