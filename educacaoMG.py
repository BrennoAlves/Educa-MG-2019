import pandas as pd
import seaborn as sns
import numpy as np
import pylab as py
import scipy.stats as stats
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import statsmodels.formula.api as sm

"""# Analise dos Dados do Enem 2019"""

#dados do enem 2019 de provas feitas em Minas Gerais
#https://drive.google.com/file/d/1uckzyjRBKPFEAp2ZN0EsV1nmHWolr93c/view?usp=sharing
enem = pd.read_csv('mg.csv',';')

#tamanho total dos dados crus
enem.shape

#Nota total igual Ciencias da Natureza, ciencias humanas, matematica, linguagem e codigos, redação / 5
notas = ['NU_NOTA_CN','NU_NOTA_CH','NU_NOTA_MT','NU_NOTA_LC','NU_NOTA_REDACAO']
enem[notas].head()

#limpando os dados tirando todos que tiraram 0 em alguma materia e todos os treineiros
enem = enem.query('NU_NOTA_CN > 0')
enem = enem.query('NU_NOTA_CH > 0')
enem = enem.query('NU_NOTA_MT > 0')
enem = enem.query('NU_NOTA_LC > 0')
enem = enem.query('NU_NOTA_REDACAO > 0')
enem = enem.query('IN_TREINEIRO == 0')
enem.shape

enem.describe()

#Gráfico de idade 
temp = enem['NU_IDADE'].value_counts().sort_index()
sns.lineplot(data=temp)

#Boxplot de idade
plt.figure(figsize=(10,10))
sns.boxplot(x='NU_IDADE', data= enem)

#Total de idade em ordem de recorrencia 
enem['NU_IDADE'].value_counts()

#criação da coluna nota total
enem['NU_NOTA_TOTAL'] = enem[notas].sum(axis=1)/5
notas = ['NU_NOTA_CN','NU_NOTA_CH','NU_NOTA_MT','NU_NOTA_LC','NU_NOTA_REDACAO', 'NU_NOTA_TOTAL']
enem[notas].head()

rendaOrdenada = enem['Q006'].unique()
rendaOrdenada.sort()
plt.figure(figsize=(8,8))
sns.boxplot(x='Q006', y='NU_NOTA_TOTAL', data= enem, order= rendaOrdenada)
plt.xlabel('Renda')
plt.ylabel('Nota Total')

sns.displot(enem["NU_NOTA_TOTAL"])
plt.xlim(0,1000)

#conferindo a normalidade da nota total 
fig, ax = plt.subplots()
stats.probplot(enem['NU_NOTA_TOTAL'], fit=True,   plot=ax)
plt.show()

#correlção das notas por materia 
correlacao = enem[notas].corr()
sns.heatmap(correlacao, cmap="Blues", annot=True )

#ordenação de registros por cidade
#Tem registro de todas as cidades de MG
enem['NO_MUNICIPIO_RESIDENCIA'].value_counts()

notaTotal = enem
notaTotal = notaTotal.drop(['NU_IDADE'], axis=1)
notaTotal = notaTotal.drop(['TP_SEXO'], axis=1)
notaTotal = notaTotal.drop(['TP_COR_RACA'], axis=1)
notaTotal = notaTotal.drop(['IN_TREINEIRO'], axis=1)
notaTotal = notaTotal.drop(['Q006'], axis=1)
notaTotal[['NU_NOTA_CN', 'NU_NOTA_CH', 'NU_NOTA_LC', 'NU_NOTA_MT', 'NU_NOTA_REDACAO', 'NU_NOTA_TOTAL']] = notaTotal[['NU_NOTA_CN', 'NU_NOTA_CH', 'NU_NOTA_LC', 'NU_NOTA_MT', 'NU_NOTA_REDACAO', 'NU_NOTA_TOTAL']].apply(pd.to_numeric)
notaTotal

notaTotal = notaTotal.groupby(['NO_MUNICIPIO_RESIDENCIA']).mean()
notaTotal

"""# IFDM MG"""

#dados do IFDM 2016 de Minas Gerais
ifdm = pd.read_csv('IFDM2016.csv',';')
ifdm.dropna
ifdm.shape

#A escala de IFDM vai de 0 a 1, multipliquei por 1000 para conseguir manipular mais facilmente 
ifdm['IFDM'] = ifdm['IFDM'].astype(float)
ifdm['IFDM'] = ifdm['IFDM']*1000
ifdm['Emprego & Renda'] = ifdm['Emprego & Renda'].astype(float)
ifdm['Emprego & Renda'] = ifdm['Emprego & Renda']*1000
ifdm['Educação'] = ifdm['Educação'].astype(float)
ifdm['Educação'] = ifdm['Educação']*1000
ifdm['Saúde'] = ifdm['Saúde'].astype(float)
ifdm['Saúde'] = ifdm['Saúde']*1000

ifdm.describe()

#Gráfico de recorrencia de IFDM
sns.distplot(ifdm['IFDM'])
plt.xlim(0,1000)

#Gráfico de recorrencia de IFDM de Educação
sns.distplot(ifdm['Educação'])
plt.xlim(0,1000)

#conferindo a normalidade do IFDM geral
fig, ax = plt.subplots()
stats.probplot(ifdm['IFDM'], fit=True,   plot=ax)
plt.show()

#conferindo a normalidade do IFDM Educação
fig, ax = plt.subplots()
stats.probplot(ifdm['Educação'], fit=True,   plot=ax)
plt.show()

#conferindo a normalidade do IFDM Saúde
fig, ax = plt.subplots()
stats.probplot(ifdm['Saúde'], fit=True,   plot=ax)
plt.show()

#conferindo a normalidade do IFDM Emprego e Renda
fig, ax = plt.subplots()
stats.probplot(ifdm['Emprego & Renda'], fit=True,   plot=ax)
plt.show()

#Mapa de calor da correlação das catergorias do IFDM
criterios = ['IFDM','Emprego & Renda',	'Educação',	'Saúde']
correlacao = ifdm[criterios].corr()
sns.heatmap(correlacao, cmap="Blues", annot=True )

"""# IDEB 2019"""

ideb = pd.read_excel('IDEB2019.xlsx')
ideb['Total'] = ideb['Total']*10
ideb

ideb.describe()

ideb.corr()

#retirando o indicador de rendimento por ser severamente proximo do indicador total
ideb = ideb.drop(['Indicador de Rendimento (P)'], axis = 1)

sns.displot(ideb)

fig, ax = plt.subplots()
stats.probplot(ideb['Total'], fit=True,   plot=ax)
plt.show()

"""# Correlação"""

cor = pd.read_excel('compilado.xlsx')
cor = cor.set_index('Nome do Município')
cor.dtypes

cor

sns.heatmap(cor.corr(), cmap="Blues", annot=True )

sns.pairplot(cor)

"""# Regressão Linear 1"""

x = cor.iloc[:, 3].values
x = x.reshape(-1, 1)
y = cor.iloc[:, 0].values
modelo = LinearRegression()
modelo.fit(x,y)
modelo.score(x,y)

previsoes = modelo.predict(x)
plt.scatter(x, y)
plt.plot(x, previsoes, color = 'red')

# modedelo 2 
modeloAjustado = sm.ols(formula= 'Enem ~ Geral + Renda + Educacao + Saude + IDEB', data = cor)
modeloTreinado = modeloAjustado.fit()
modeloTreinado.summary()