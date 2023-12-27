import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

year = 2015
data = pd.read_excel("D:/КПІ/3-курс/Data Science/project/DS_IndexProsperity/SensitivityAnalysis_Result/" + f"{year}.xlsx")

selected17 = data.iloc[:, 3:20].columns.tolist() + [data.columns[25]]
data17 = data[selected17]
data5 = data.iloc[:, 20:26]

correlation_matrix17 = data17.corr()
correlation_matrix5 = data5.corr()

plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix5, annot=True, cmap="coolwarm", fmt=".2f", linewidths=.5)
plt.title("Кореляційна матриця")
plt.show()

plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix17, annot=True, cmap="coolwarm", fmt=".2f", linewidths=.5)
plt.title("Кореляційна матриця")
plt.show()

