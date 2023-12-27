import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from SALib.sample import saltelli
from SALib.analyze import sobol

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

param = saltelli.sample(problem, 1024)
si = sobol.analyze(problem, model(param))
print('Sobol First-Order Indices:', si['S1'])  # Показники чутливості за Соболем першого порядку
print('Sobol Total-Order Indices:', si['ST'])  # Загальні показники чутливості за Соболем

namePlot = "Sobol index for 5 indexs"
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
ax.set_xticklabels(problem['names'])
ax.legend([s1, st], ['Sobol First-Order Indices', 'Sobol Total-Order Indices'])

plt.savefig('D:/КПІ/3-курс/Data Science/project/DS_IndexProsperity/SensitivityAnalysis_Result/' + f'{namePlot}.png')
plt.show()

"""""
def createDataA(data, N):
    return data.sample(n = N, replace = True, random_state=2).values

def createDataB(data, N):
    return data.sample(n = N, replace = True, random_state=3).values

def createDataAB(dataA, dataB, variableIndexToFix):
    dataBwithA = dataB.copy()
    dataBwithA[:, variableIndexToFix] = dataA[:, variableIndexToFix]
    return dataBwithA

def sobolIndice1stAndTotalOrder(model, variableIndex, dataA, dataB):

    dataBwithA = createDataAB(dataA, dataB, variableIndex)

    N = len(dataA)

    yA = model(dataA)
    yAB = model(dataBwithA)
    yB = model(dataB)

    num1stOrder = N*np.sum(np.multiply(yA, yAB)) - (np.sum(yA)*np.sum(yAB))
    numTotal = N*np.sum(np.multiply(yB, yAB)) - (np.sum(yA)**2)
    deno = N*np.sum(yA**2) - (np.sum(yA))**2

    return np.round(num1stOrder/deno, 3), np.round((1 - (numTotal/deno)), 3)

def plotResults(X, namePlot, model, dataA, dataB):

    stOrder = []
    total = []

    for i in range((X.shape[1])):
        results = sobolIndice1stAndTotalOrder(model, i, dataA, dataB)
        stOrder.append(results[0])
        total.append(results[1])


    df_result = pd.DataFrame({
        "variable" : list(X.columns),
        "1st order" : stOrder,
        "other order" : [tot - first for tot, first in zip(total, stOrder)]
    })

    bar_width = 0.35
    fig, ax = plt.subplots()
    ax.bar(np.arange(len(df_result)), df_result["1st order"], width=bar_width, label='1st order')
    ax.bar(np.arange(len(df_result)) + bar_width, df_result["other order"], width=bar_width, label='other order')
    ax.set_xticks(np.arange(len(df_result)) + bar_width / 2)
    ax.set_xticklabels(df_result["variable"])

    ax.set_xlabel('Indices', fontsize=5)
    plt.savefig('D:/КПІ/3-курс/Data Science/project/DS_IndexProsperity/SensitivityAnalysis_Result/' + f'{namePlot}.png')
    plt.show()
dataA = createDataA(Indexs5, 6500)
dataB = createDataB(Indexs5, 6500)
plotResults(Indexs5, "Sobol index for 5 indexs", model, dataA, dataB)
"""
