import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from SALib.sample import saltelli
from SALib.analyze import delta
from sklearn.impute import KNNImputer
from sklearn.cluster import KMeans
from kneed import KneeLocator
from factor_analyzer import FactorAnalyzer
from factor_analyzer.factor_analyzer import calculate_kmo

def model(data):
    # factorizing security

    fa_s = FactorAnalyzer(rotation='varimax')
    fa_s.fit(data[:, 10:16 + 1])
    # processing loadings
    loadings = pd.DataFrame(fa_s.loadings_)
    loadings = loadings.apply(lambda x: x * x)
    loadings = loadings.apply(lambda column: column / column.sum(), axis=0)

    expl_var = fa_s.get_factor_variance()[0]

    # computing sub indexes and index
    Education = (data[:, 0] * 0.5 + (data[:, 1] * 0.5 / 3 + data[:, 2] * 0.5 / 3 + data[:, 3] * 0.5 / 3))
    Economic = data[:, 4:6 + 1].mean()
    Health = (data[:, 7] * 0.3 + data[:, 8] * 0.7)
    Environment = data[:, 9].mean()

    n = data.shape[0]
    arr = np.zeros((n, len(loadings.columns)))

    for i in loadings.columns:
        arr[:, i] = 0
        index = 0
        for weight in loadings[i]:
            if weight > 0.1:
                arr[:, i] += weight * data[:, 10 + index]
            index += 1

    Safety = 0
    for i in loadings.columns:
        coef = expl_var[i] / expl_var.sum()
        Safety += arr[:, i] * coef

    coef_for_environment = 0.1
    coef_for_others = (1 - coef_for_environment) / (5 - 1)

    y = Environment * coef_for_environment + coef_for_others * (Economic + Education + Health + Safety)

    return y

year = 2015
data = pd.read_excel("D:/КПІ/3-курс/Data Science/project/DS_IndexProsperity/SensitivityAnalysis_Result/" + f"{year}.xlsx")
Indexs17 = data.iloc[:, 3:20]

minValue = Indexs17.min() #0
maxValue = Indexs17.max() #1

problem = {
    'num_vars': 17,  #Кількість параметрів
    'names': ['Unemployment', 'Secondary\nschool', 'Primary\nschool', 'Tertiary\nschool',
              'WDI', 'Access\nto\nelectricity', 'GDP',
              'Mortality', 'Life\nexpectancy',
              'CO2\nemissions',
              'Democracy\nindex', 'Control\nof\nCorruption', 'Government\nEffectiveness' ,'Political Stability\nand\nAbsence of Terrorism',
              'Regulatory\nQuality', 'Rule\nof\nLaw', 'Voice\nand\nAccountability'],  #Назви параметрів
    'bounds': [(minValue.iloc[0], maxValue.iloc[0]),
               (minValue.iloc[1], maxValue.iloc[1]),
               (minValue.iloc[2], maxValue.iloc[2]),
               (minValue.iloc[3], maxValue.iloc[3]),
               (minValue.iloc[4], maxValue.iloc[4]),
               (minValue.iloc[5], maxValue.iloc[5]),
               (minValue.iloc[6], maxValue.iloc[6]),
                (minValue.iloc[7], maxValue.iloc[7]),
                (minValue.iloc[8], maxValue.iloc[8]),
                (minValue.iloc[9], maxValue.iloc[9]),
                (minValue.iloc[10], maxValue.iloc[10]),
                (minValue.iloc[11], maxValue.iloc[11]),
                (minValue.iloc[12], maxValue.iloc[12]),
                (minValue.iloc[13], maxValue.iloc[13]),
                (minValue.iloc[14], maxValue.iloc[14]),
                (minValue.iloc[15], maxValue.iloc[15]),
                (minValue.iloc[16], maxValue.iloc[16])] # Межі для кожного параметра
}

param = Indexs17.sample(n = 2000, replace = True, random_state=2).values

TAI = delta.analyze(problem, param, model(param))

print('Total-Effect Indices:', TAI['delta'])

namePlot = "Total-Effect indices for 17 indexs"
df_result = pd.DataFrame({
    "indices" :  problem['names'],
    "Total-Effect Indices" : TAI['delta'],
})

ax = df_result.plot(x="indices", kind="bar", stacked=True, rot=0, title="Total-Effect Indices", fontsize=7)
plt.savefig('D:/КПІ/3-курс/Data Science/project/DS_IndexProsperity/SensitivityAnalysis_Result/' + f'{namePlot}.png')
plt.show()

"""""
'names': ['Unemployment', 'Secondary\nschool', 'Primary\nschool', 'Tertiary\nschool',
              'WDI', 'Electricity', 'GDP',
              'Mortality', 'Life\nexpectancy',
              'CO2\nemissions',
              'Democracy\nindex', 'Control\nof\nCorruption', 'Political\nStability\nand Absence of Terrorism',
              'Regulatory\nQuality', 'Rule of Law', 'Voice\nand Accountability'],  #Назви параметрів
"""