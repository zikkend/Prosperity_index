# Перевірка на чутливість для 2015 року

Дані з [2015](https://github.com/PolnikovaPolina/DS_IndexProsperity/blob/main/SensitivityAnalysis_Result/2015.xlsx) року. Варто зазначити, що будемо працювати з біблотекою SALib, тому завантажимо її за допомогою команди: 
```R
pip install SALib
```
------
# Показник чутливості за Соболем

Тепер будемо обчислювати показники чутловості за Соболем першого порядку (S1) та загальний (ST). Варто зазначити, що **показник чутловості першого порядку** вимірює вплив окремого параметра на варіацію вихідної змінної. Насамперед **показник загальної чутливості** вимірює вплив конкретного параметра, а також всіх можливих взаємодій між параметрами на варіацію вихідної змінної. 

Отже, спочатку обчислимо показники чутливості **для 5 загальних індексів**. Для цього нам варто написати модель обчислення рівня добробуту для країн за допомогою цих індексів: економіка, екологія, освіта, здоров'я та безпека, які формуються у таблицю data згідно їх порядку у [загальній таблиці показників](https://github.com/PolnikovaPolina/DS_IndexProsperity/blob/main/SensitivityAnalysis_Result/2015.xlsx).
```R
def model(data):
    coef_for_environment = 0.1
    coef_for_others = (1 - coef_for_environment) / (5 - 1)

    y = (data[:, 1] * coef_for_environment + coef_for_others *
       (data[:, 0] + data[:, 2] + data[:, 3] + data[:, 4]))

    return y
```
Та генеруємо Собольський плану чутливості за допомогою модуля saltelli в бібліотеці SALib, щоб визначити, як різні вхідні параметри впливають на вихідні значення моделі.
```R
param = saltelli.sample(problem, 1024)
```
Отже, отримали графік:

![](https://github.com/PolnikovaPolina/DS_IndexProsperity/blob/main/SensitivityAnalysis_Result/Sobol%20index%20for%205%20indexs.png)

Та результати:
```
Sobol First-Order Indices: [0.2376385  0.07169183  0.18829598  0.32035063  0.18169995]
Sobol Total-Order Indices: [0.23794709  0.07173138  0.18833724  0.32017976  0.1815501]
```
Бачимо, що найзначущій показник для рівня добробуту це здоров'я, він пояснює близько 32% дисперсії рівня добробуту, далі йде економіка 24%, потім освіта 19%, потім безпека 18%, а потім екологія 7%.

Тепер обчислимо показники чутливості **для 17 індексів**. Для цього нам варто написати вже іншу модель обчислення рівня добробуту для країн за допомогою вже індексів: ```Unemployment, total, School enrollment, secondary, School enrollment, primary, School enrollment, tertiary, World Development Indicators, Access to electricity, GDP per capita, Mortality rate, under-5, Life expectancy at birth, CO2 emissions, Democracy index, Control of Corruption: Estimate, Government Effectiveness: Estimate, Political Stability and Absence of Violence/Terrorism: Estimate, Regulatory Quality: Estimate, Rule of Law: Estimate, Voice and Accountability: Estimate```, які формуються у таблицю data згідно їх порядку у [загальній таблиці показників](https://github.com/PolnikovaPolina/DS_IndexProsperity/blob/main/SensitivityAnalysis_Result/2015.xlsx).
```R
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
```
Отже, отримали графік:

![](https://github.com/PolnikovaPolina/DS_IndexProsperity/blob/main/SensitivityAnalysis_Result/Sobol%20index%20for%2017%20indexs.png)

Та результати:
```
Sobol First-Order Indices: [1.99182455e-01  2.23089951e-02  2.23925425e-02 2.17102504e-02  0.00000000e+00  0.00000000e+00  0.00000000e+00  7.09655056e-02  3.91360315e-01  0 00000000e+00  1.34635081e-01  6.74062084e-03  0.00000000e+00  3.80730745e-04  1.64709262e-03  1.28243329e-01  0.00000000e+00]

Sobol Total-Order Indices: [0.19930098  0.0220549  0.02214757  0.02215733 0.0  0.0  0.0  0.07176023  0.3912947  0.0  0.13449125  0.0677182  0.0  0.00041103  0.00164111  0.00172743  0.0]
```
Бачимо, що найзначущій показник для рівня добробуту це тривалість життя, він пояснює близько 42% дисперсії рівня добробуту, далі йде безробіття 19%, потім демократичний показник 13%, потім контроль над корупцією 7%, потім сметрність 6%, потім показники зарахування до молодшої, середньої та старшої школи по 2%, а потім показники регулярної якості безпеки та верховенства права. Причому інші показники не впливають на рівень добробуту країн.

Зауважимо, що для всіх підіндексів показник чутловості першого порядку та показник загальної чутливості однакові, але для підіндекса тривалість життя загальна чутливість (0,4) менша, ніж чутливість першого порядку (0,42). Це означає, що показник тривалості життя разом із взаємодіями з іншими параметрами, пояснює близько 40% дисперсії рівня добробуту в моделі. Натомість сам показник тривалості життя внесе приблизно 42% в дисперсію рівня добробуту, не враховуючи взаємодій з іншими параметрами.

# Висновок за показники чутливості за Соболем

Найважливіший показник на формування рівня добробуту для країн серед 5 загальний індексів це - показник здоров'я, а найменшважливіший - це показник екології.
Найважливіший показник на формування рівня добробуту для країн серед 17 індексів це - показник тривалості життя людини, а найменш важливіші - це показники розвитку світу, ВВП, доступу до електроенергії, викидів СО2, ефективності урядів, політичної стабільності та відсутності тероризму, права голосу. Але варто зазначити, що показник чутливості за Соболем має деякі погрішності, бо оцінюють тільки прямий вплив кожного параметра на вихід, враховують тільки лінійні ефекти. Тому доречно було обчислити і інший показниу чутловості Total-Effect Index (TAI), який як раз і буде враховувати як прямий вплив (чутливість першого порядку), так і непрямий вплив через взаємодію між вхідними факторами. 

# Показник чутливості за TAI

Отже, спочатку знову обчислимо показники чутливості **для 5 загальних індексів**. Для цього нам варто використати [минулу модель](#Показник-чутливості-за-Соболем) обчислення рівня добробуту для країн за допомогою індексів: економіка, екологія, освіта, здоров'я та безпека, які формуються у таблицю data згідно їх порядку у [загальній таблиці показників](https://github.com/PolnikovaPolina/DS_IndexProsperity/blob/main/SensitivityAnalysis_Result/2015.xlsx).
Однак вже будемо генерувати плану чутливості на основі знайдених індексів у [загальній таблиці показників](https://github.com/PolnikovaPolina/DS_IndexProsperity/blob/main/SensitivityAnalysis_Result/2015.xlsx), де вибірка формується на основі випадкового вибору рядків Indexs5 із заміщенням. Натомість розмір вибірки буде 5000 зразків.
```R
param = Indexs5.sample(n = 5000, replace = True, random_state=2).values
```
Отже, отримали графік:

![](https://github.com/PolnikovaPolina/DS_IndexProsperity/blob/main/SensitivityAnalysis_Result/Total-Effect%20indices%20for%205%20indexs.png)

Та результати:
```
Total-Effect Indices: [0.42739223 0.2316124  0.37687858 0.25895057 0.36819535]
```
Бачимо, що найзначущій показник для рівня добробуту це економіка, яка впливає на 42% на зміну вихідного показника, враховуючи взаємодії з іншими параметрами, далі йде освіта 38%, потім безпека 36%, потім здоров'я 25%, а потім екологія 23%.

Тепер обчислимо показники чутливості **для 17 індексів**. Для цього нам потрібно використати [минулу модель](#Показник-чутливості-за-Соболем) обчислення рівня добробуту для країн за допомогою вже індексів: ```Unemployment, total, School enrollment, secondary, School enrollment, primary, School enrollment, tertiary, World Development Indicators, Access to electricity, GDP per capita, Mortality rate, under-5, Life expectancy at birth, CO2 emissions, Democracy index, Control of Corruption: Estimate, Government Effectiveness: Estimate, Political Stability and Absence of Violence/Terrorism: Estimate, Regulatory Quality: Estimate, Rule of Law: Estimate, Voice and Accountability: Estimate```, які формуються у таблицю data згідно їх порядку у [загальній таблиці показників](https://github.com/PolnikovaPolina/DS_IndexProsperity/blob/main/SensitivityAnalysis_Result/2015.xlsx).

Отже, отримали графік:

![](https://github.com/PolnikovaPolina/DS_IndexProsperity/blob/main/SensitivityAnalysis_Result/Total-Effect%20indices%20for%2017%20indexs.png)

Та результати:
```
Total-Effect Indices: [0.18975642  0.27695155  0.21760007  0.24906309  0.15110483  0.26012162  0.27744814  0.26083286  0.27864177  0.23696029  0.24426854  0.27191441  0.31364412  0.25351981  0.2667406  0.2677794  0.26060317]
```
Бачимо, що найзначущій показник для рівня добробуту це ефективність уряду, яка впливає на 31% на зміну вихідного показника, враховуючи взаємодії з іншими параметрами, найменш значуща це показник безробіття (19%) та рівня розвитку світу (15%).

# Висновок за показники чутливості за TAI

Найважливіший показник на формування рівня добробуту для країн серед 5 загальний індексів це - показник економіки, а найменшважливіший - це показник екології.
Найважливіший показник на формування рівня добробуту для країн серед 17 індексів це - показник ефективності уряду, а найменш важливіші - це показники розвитку світу.

# Загальний висновок
Отримані дані свідчать, що найменш важливий для формування самого індексу добробуту це екологія, але видалити його нераціонально, оскільки хоч він і наймеш важливий, але за показником чутливості TAI він впливає на 23% на зміну вихідного показника, враховуючи взаємодії з іншими параметрами. Аналогічно стосується і рівня розвитку. При цьому варто зазначити, що показник чутливості за TAI набагато точніший, ніж показник чутливості за Соболем, оскільки показник за Соболем показав, що аж 7 індексів з 17 взагалі ніяк не впливають на дисперсію індекса добробуту, і тільки 4 індекси не мінімально впливають на неї. Хоча аналіз за показником чутливості TAI показав, що всі індексти так або інакше впливають на формування добробуту країн. Оскільки TAI враховує усі можливі взаємодії між параметрами, що дозволяє отримати комплексний погляд на їхній вплив. Натомість показники чутливості за Соболем першого порядку враховує обмежені врахуванням тільки лінійних взаємодій.
 
# Додаток 
[Код на аналіз 5 індексів на рівень добробуту країн за допомогою показників чутливості за Соболем](https://github.com/PolnikovaPolina/DS_IndexProsperity/blob/main/SobolIndex5.py)

[Код на аналіз 17 підіндексів на рівень добробуту країн за допомогою показників чутливості за Соболем](https://github.com/PolnikovaPolina/DS_IndexProsperity/blob/main/SobolIndex17.py) 

[Код на аналіз 5 індексів на рівень добробуту країн за допомогою показників чутливості за TAI](https://github.com/PolnikovaPolina/DS_IndexProsperity/blob/main/TotalEffectIndex5.py) 

[Код на аналіз 17 підіндексів на рівень добробуту країн за допомогою показників чутливості за TAI](https://github.com/PolnikovaPolina/DS_IndexProsperity/blob/main/TotalEffectIndex17.py)

# Список джерел:
 * https://towardsdatascience.com/sobol-indices-to-measure-feature-importance-54cedc3281bc
 * https://salib.readthedocs.io/en/latest/_modules/SALib/analyze/sobol.html
 * https://salib.readthedocs.io/en/latest/user_guide/basics.html#an-example

Зауважимо, що для побудлви графіків було перейменовані індекси:
* Unemployment, total (% of total labor force) (modeled ILO estimate) - Unemployment
* School enrollment, secondary (% net) - Secondary school
* School enrollment, primary (% net) - Primary school
* School enrollment, tertiary (% gross) - Tertiary school
* World Development Indicators - WDI
* Access to electricity (% of population) - Access to electricity
* GDP per capita (current US$) - GDP
* Mortality rate, under-5 (per 1,000 live births) - Mortality
* Life expectancy at birth, total(years) - Life expectancy
* CO2 emissions (metric tons per capita) - CO2 emissions
* Control of Corruption: Estimate - Control of Corruption:
* Government Effectiveness: Estimate - Government Effectiveness
* Political Stability and Absence of Violence/Terrorism: Estimate - Political * Stability
* Regulatory Quality: Estimate - Regulatory Quality
* Rule of Law: Estimate - Rule of Law
* Voice and Accountability: Estimate - Voice and Accountability



