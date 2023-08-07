import pandas as pd
from scipy.special import gamma, psi, polygamma
import scipy.stats as stats

xl = pd.ExcelFile("./adatok2.xlsx")
xl.sheet_names

df = xl.parse("Sheet2")
df.head()


fit1 = stats.gamma.fit(df['k'], *[1], method='mm')
fit2 = stats.gamma.fit(df['theta'], loc=0)


xl.close()