import pandas as pd
import numpy as np
from SALib.sample import saltelli
from SALib.analyze import sobol
import matplotlib.pyplot as plt
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

param = saltelli.sample(problem, 1024)
si = sobol.analyze(problem, model(param))
print('Sobol First-Order Indices:', si['S1'])  # Показники чутливості за Соболем першого порядку
print('Sobol Total-Order Indices:', si['ST'])  # Загальні показники чутливості за Соболем

namePlot = "Sobol index for 17 indexs"
df_result = pd.DataFrame({
    "indices" :  problem['names'],
    "Sobol First-Order Indices" : si['S1'],
    "Sobol Total-Order Indices" : si['ST']
})

bar_width = 0.35
fig, ax = plt.subplots()
s1 = ax.bar(np.arange(len(problem['names'])), si['S1'], width=bar_width, label='Sobol First-Order Indices')
st = ax.bar(np.arange(len(problem['names'])) + bar_width, si['ST'], width=bar_width, label='Sobol Total-Order Indices')
ax.set_xticks(np.arange(len(problem['names'])) + bar_width / 2)
ax.set_xticklabels(problem['names'], fontsize=6.5)
ax.legend([s1, st], ['Sobol First-Order Indices', 'Sobol Total-Order Indices'])
plt.savefig('D:/КПІ/3-курс/Data Science/project/DS_IndexProsperity/SensitivityAnalysis_Result/' + f'{namePlot}.png')
plt.show()

"""""
new_column_names = ['Unemployment', 'Secondary\nschool', 'Primary\nschool', 'Tertiary\nschool',
              'WDI', 'Access\nto\nelectricity', 'GDP',
              'Mortality', 'Life\nexpectancy',
              'CO2\nemissions',
              'Democracy\nindex', 'Control\nof\nCorruption', 'Government\nEffectiveness' ,'Political\nStability\nand\nAbsence of Terrorism',
              'Regulatory\nQuality', 'Rule of Law', 'Voice\nand\nAccountability']

Indexs17 = Indexs17.rename(columns=dict(zip(Indexs17.columns, new_column_names)))

dataA = createDataA(Indexs17, 9000)
dataB = createDataB(Indexs17, 9000)
plotResults(Indexs17, "Sobol index for 17 indexs", modelFor17Indexs, dataA, dataB)

"""

