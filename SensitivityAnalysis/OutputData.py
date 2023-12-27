import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.cluster import KMeans
from kneed import KneeLocator
from factor_analyzer import FactorAnalyzer
from factor_analyzer.factor_analyzer import calculate_kmo

imputer = KNNImputer(n_neighbors=5)

# reversed indicators
reversed = ['Unemployment, total (% of total labor force) (modeled ILO estimate)',
            'Mortality rate, under-5 (per 1,000 live births)',
            'CO2 emissions (metric tons per capita)']
# sub indexes
education = [ 'Unemployment, total (% of total labor force) (modeled ILO estimate)',
              'School enrollment, secondary (% net)',
              'School enrollment, primary (% net)',
              'School enrollment, tertiary (% gross)']
economic = ['World Development Indicators',
            'Access to electricity (% of population)',
            'GDP per capita (current US$)']
health = ['Mortality rate, under-5 (per 1,000 live births)',
          'Life expectancy at birth, total(years)']
environment = ['CO2 emissions (metric tons per capita)']
security = ['Democracy index',
            'Control of Corruption: Estimate',
            'Government Effectiveness: Estimate',
            'Political Stability and Absence of Violence/Terrorism: Estimate',
            'Regulatory Quality: Estimate',
            'Rule of Law: Estimate',
            'Voice and Accountability: Estimate']
list_of_sub_indexes = [education, economic, environment, health, security]
sub_indexes = ['Education', 'Economic', 'Health', 'Environment', 'Safety']

list_of_dfs = []
def convert_separate(year):
    data = pd.read_csv('D:/КПІ/3-курс/Data Science/project/DS_IndexProsperity/SensitivityAnalysis_Data/' + f'DS_Index - {year}.csv')
    # safety index cleaning
    safety = pd.read_excel("D:/КПІ/3-курс/Data Science/project/DS_IndexProsperity/SensitivityAnalysis_Data/Safety.xlsx")
    names = ['Control of Corruption: Estimate', 'Government Effectiveness: Estimate', 'Political Stability and Absence of Violence/Terrorism: Estimate', 'Regulatory Quality: Estimate', 'Rule of Law: Estimate', 'Voice and Accountability: Estimate']
    safety = safety[safety['Series Name'].isin(names)]
    # merging data
    for i in range(len(names)):
        s = safety[safety['Series Name'] == names[i]][['Country Code', f'{year} [YR{year}]']]
        col = s.columns.tolist()
        col[-1] = names[i]
        s.columns = col
        data = data.merge(s, on=['Country Code'])

    # deleting unnecessary information
    to_delete = ['Rail lines (total route-km)', 'Net migration', 'Annual freshwater withdrawals, total (% of internal resources)']
    data.drop(labels=to_delete, inplace=True, axis=1)

    # converting to float
    data.replace('..', np.nan, inplace=True)
    cols = data.columns
    cols = cols[2:]
    data[cols] = data[cols].astype(str)
    data = data.apply(lambda x: x.str.replace(',','.'))
    data[cols] = data[cols].astype(float)

    #remove rows with more than 3 NaN
    data = data[data.isnull().sum(axis=1) < 3]
    data.reset_index(inplace=True, drop=True)

    # filling missing values with KNN, k = 5
    data[cols] = pd.DataFrame(imputer.fit_transform(data[cols]),columns = cols)

    # normalizing data
    data[cols]=(data[cols]-data[cols].min())/(data[cols].max()-data[cols].min())
    for i in reversed:
        data[i] = 1 - data[i]

    # factorizing security
    fa_s = FactorAnalyzer(rotation='varimax')
    fa_s.fit(data[security])

    # processing loadings
    loadings = pd.DataFrame(fa_s.loadings_)
    loadings = loadings.apply(lambda x: x*x)
    loadings = loadings.apply(lambda column: column/column.sum(), axis=0)

    # creating temporary intermediate indicators
    for i in loadings.columns:
        data[str(i)] = 0
        index = 0
        for weight in loadings[i]:
            if weight > 0.1:
                data[str(i)] += weight*data[security[index]]
            index += 1
    expl_var = fa_s.get_factor_variance()[0]

    # computing sub indexes and index
    data['Economic'] = data[economic].mean(axis=1)
    data['Environment'] = data[environment].mean(axis=1)
    data['Education'] = (data[education[0]]*0.5 + (data[education[1]]*0.5/3 + data[education[2]]*0.5/3 + data[education[3]]*0.5/3))
    data['Health'] = (data[health[0]]*0.3 + data[health[1]]*0.7)
    data['Safety'] = 0
    for i in loadings.columns:
        coef = expl_var[i] / expl_var.sum()
        data['Safety'] += data[str(i)] * coef

    # drop temporary indicators
    data.drop(labels = [str(i) for i in loadings.columns], axis=1, inplace=True)

    coef_for_environment = 0.1
    coef_for_others = (1 - coef_for_environment)/(len(list_of_sub_indexes)-1)

    data['Prosperity index'] = data['Environment'] * coef_for_environment + coef_for_others * (data['Economic'] + data['Education'] + data['Health'] + data['Safety'])

    #loading to excel normalized version with indexes
    data.to_excel("D:/КПІ/3-курс/Data Science/project/DS_IndexProsperity/SensitivityAnalysis_Result/" + f"{year}.xlsx")

    data.sort_values('Prosperity index', inplace=True, ascending=False)

    #loading result to excel
    res = data[['Country Name', 'Country Code'] + sub_indexes + ['Prosperity index']].reset_index(drop=True)
    res.index += 1
    res.reset_index(names="Rank", inplace = True)
    res.index += 1
    res.to_excel("D:/КПІ/3-курс/Data Science/project/DS_IndexProsperity/SensitivityAnalysis_Result/" + f"{year}_result.xlsx")

    data.reset_index(inplace=True, drop=True)
    data.index += 1
    data.reset_index(inplace=True, names='Rank')
    data.index += 1
    data = pd.concat([data], keys=[f'{year}'], names=['Year'])
    list_of_dfs.append(data)

# Testing for 2015
convert_separate(2015)