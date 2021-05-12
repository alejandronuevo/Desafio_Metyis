import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

#Importamos los datos del dataset que vamos a utilizar
dataset = pd.read_csv('C:/Users/casa/Desktop/caso entrevista/movies.csv',encoding='latin1')

#Consultamos la informacion de los datos para ver el numero de registros y los tipos de datos
dataset.info()

#Eliminamos los parametros que no se han de utilizar que estan indicados en el documento 
dataset_limpio = dataset.drop(['director','star','writer'],axis=1)

#Confirmamos que la informacion se ha eliminado correctamente
dataset_limpio.info()

#Analizamos los valores de cada uno de los parametros para revisar si hubiese que realizar alguna conversion en el dataset
descripcion=dataset_limpio.describe(include="all")

#En el paso anterior se puede observar que hay registros en el campo budget que son 0, como esto puede hacer que el analisis 
#no sea tan preciso se decide eliminar estos registros del estudio posterior
dataset_limpio[dataset_limpio['budget']==0.0]= np.nan
dataset_limpio=dataset_limpio.dropna()

#Comprobamos que el minimo presupuesto ya no es 0
descripcion_sin_0=dataset_limpio.describe(include="all")

#Separamos dentro de la variable released el mes ya que es el dato que nos puede dar informacion posteriormente y el año lo tenemos en year
dataset_limpio['released'] = pd.to_datetime(dataset_limpio['released'])
dataset_limpio['released']=(dataset_limpio['released'].dt.month).astype('object')

#Creamos una nueva variable en un nuevo dataset que sea la relacion entre ingresos y gastos para ver de forma mas clara el beneficio
dataset_limpio_rbf=dataset_limpio.copy()
dataset_limpio_rbf['ratio_beneficio_coste'] = dataset_limpio_rbf['gross'] / dataset_limpio_rbf['budget']

p_elem = dataset_limpio_rbf.head()

#Esta nueva variable contiene outliers que ensucian mucho los datos con lo que se procede a eliminar los mismos 
def eliminar_outliers(Df, col_name):
  Q1 = Df[col_name].quantile(.25)
  Q3 = Df[col_name].quantile(.75)
  IQR = Q3 - Q1
  limite_inf = Q1 - 1.5*IQR
  limite_sup = Q3 + 1.5*IQR
  
  return Df[(Df[col_name]>=limite_inf) & (Df[col_name]<=limite_sup)]

dataset_limpio_rbf = eliminar_outliers(dataset_limpio_rbf, 'ratio_beneficio_coste')
#dataset_limpio_rbf['ratio_beneficio_coste'].plot(kind='box')

#Analizamos primero las relaciones entre las variables numericas originales del dataset 
sns.heatmap(dataset_limpio.corr(), annot = True, linewidths=.5)
plt.title('Correlaciones variables', fontsize = 15)

#Dibujamos aquellas que parecen tener mas relacion entre si
sns.pairplot(dataset_limpio[['votes','budget','gross']])

#Realizamos el mismo analisis con la nueva variable creada
sns.pairplot(dataset_limpio_rbf[['ratio_beneficio_coste','budget','gross']])

#Analizamos las variables categoricas que usaremos en posteriores visualizaciones
peliculas_x_genero=dataset_limpio['genre'].value_counts()
peliculas_x_rating=dataset_limpio['rating'].value_counts()

#Como hay elementos con muy pocas apariciones los unimos en un grupo generico
def group_low_freq_cats(Df, col_name, threshold=0.01, name='Resto'):
  df = Df.copy()
  cat_freq = df[col_name].value_counts()
  cat_low_freq = cat_freq[cat_freq/cat_freq.sum() <= threshold].index
  df.loc[df[col_name].isin(cat_low_freq),col_name]='Resto'
  return df

dataset_limpio = group_low_freq_cats(dataset_limpio,'genre',threshold=0.01) 
dataset_limpio = group_low_freq_cats(dataset_limpio,'rating',threshold=0.01)
dataset_limpio_rbf = group_low_freq_cats(dataset_limpio_rbf,'genre',threshold=0.01) 
dataset_limpio_rbf = group_low_freq_cats(dataset_limpio_rbf,'rating',threshold=0.01)

#Generamos algunas visualizaciones con estos datos ya limpios en el dataset original y modificado
sns.catplot(x="budget", y="genre", kind="box", data=dataset_limpio)
plt.title('Presupuesto vs genero', fontsize = 15)
sns.catplot(x="gross", y="genre", kind="box", data=dataset_limpio)
plt.title('Recaudación vs genero', fontsize = 15)
sns.catplot(x="year", y="rating", kind="box", data=dataset_limpio)
plt.title('Año vs rating', fontsize = 15)
sns.catplot(x="runtime", y="genre", kind="box", data=dataset_limpio)
plt.title('Duración vs genero', fontsize = 15)
sns.catplot(y="genre",  kind="count", data=dataset_limpio)
plt.title('Número de elementos por genero', fontsize = 15)

#sd se refiere a la desviacion estandar, en los casos sin este parametro se usara el intervalo de confianza del 95%
sns.relplot(x="score", y="gross", kind="line", ci="sd", data=dataset_limpio)
plt.title('Nota vs recaudación', fontsize = 15)
sns.relplot(x="score", y="gross", kind="line", data=dataset_limpio)
plt.title('Nota vs recaudación', fontsize = 15)
sns.relplot(x="gross", y="votes", kind="line", ci="sd", data=dataset_limpio)
plt.title('Recaudación vs número votos', fontsize = 15)
sns.relplot(x="gross", y="votes", kind="line", data=dataset_limpio)
plt.title('Recaudación vs número votos', fontsize = 15)
sns.relplot(x="score", y="runtime", kind="line", data=dataset_limpio)
plt.title('Nota vs duración', fontsize = 15)
sns.relplot(x="score", y="runtime", kind="line", ci="sd", data=dataset_limpio)
plt.title('Nota vs duración', fontsize = 15)
sns.relplot(x="budget", y="gross", kind="line", ci="sd", data=dataset_limpio)
plt.title('Presupuesto vs recaudación', fontsize = 15)
sns.relplot(x="budget", y="gross", kind="line", data=dataset_limpio)
plt.title('Presupuesto vs recaudación', fontsize = 15)

sns.catplot(x="ratio_beneficio_coste", y="genre", kind="box", data=dataset_limpio_rbf)
plt.title('Ratio beneficio coste vs genero', fontsize = 15)
sns.catplot(x="ratio_beneficio_coste", y="rating", kind="box", data=dataset_limpio_rbf)
plt.title('Ratio beneficio coste vs rating', fontsize = 15)
sns.catplot(x="released", y="ratio_beneficio_coste", kind="box", data=dataset_limpio_rbf)
plt.title('Mes extreno vs Ratio beneficio coste', fontsize = 15)

sns.relplot(x="score", y="ratio_beneficio_coste", kind="line", ci="sd", data=dataset_limpio_rbf)
plt.title('Nota vs Ratio beneficio coste', fontsize = 15)
sns.relplot(x="score", y="ratio_beneficio_coste", kind="line", data=dataset_limpio_rbf)
plt.title('Nota vs Ratio beneficio coste', fontsize = 15)
sns.relplot(x="runtime", y="ratio_beneficio_coste", kind="line", data=dataset_limpio_rbf)
plt.title('Duración vs Ratio beneficio coste', fontsize = 15)
sns.relplot(x="runtime", y="ratio_beneficio_coste", kind="line", ci="sd", data=dataset_limpio_rbf)
plt.title('Duración vs Ratio beneficio coste', fontsize = 15)
sns.relplot(x="gross", y="ratio_beneficio_coste", kind="line", ci="sd", data=dataset_limpio_rbf)
plt.title('Recaudación vs Ratio beneficio coste', fontsize = 15)
sns.relplot(x="gross", y="ratio_beneficio_coste", kind="line", data=dataset_limpio_rbf)
plt.title('Recaudación vs Ratio beneficio coste', fontsize = 15)
sns.relplot(x="budget", y="ratio_beneficio_coste", kind="line", ci="sd", data=dataset_limpio_rbf)
plt.title('Presupuesto vs Ratio beneficio coste', fontsize = 15)
sns.relplot(x="budget", y="ratio_beneficio_coste", kind="line", data=dataset_limpio_rbf)
plt.title('Presupuesto vs Ratio beneficio coste', fontsize = 15)
sns.relplot(x="votes", y="ratio_beneficio_coste", kind="line", ci="sd", data=dataset_limpio_rbf)
plt.title('Número votos vs ratio_beneficio_coste', fontsize = 15)
sns.relplot(x="votes", y="ratio_beneficio_coste", kind="line", data=dataset_limpio_rbf)
plt.title('Número votos vs ratio_beneficio_coste', fontsize = 15)

#Para buscar mas informacion con los datos que tenemos realizamos una division del dataset en funcion del presupuesto
bajo_presupuesto = dataset_limpio_rbf[(dataset_limpio_rbf['budget']<=dataset_limpio_rbf['budget'].quantile(.25))]
b_pre=bajo_presupuesto.describe(include="all")
medio_presupuesto = dataset_limpio_rbf[(dataset_limpio_rbf['budget']>dataset_limpio_rbf['budget'].quantile(.25)) & (dataset_limpio_rbf['budget']<=dataset_limpio_rbf['budget'].quantile(.50))]
m_pre=medio_presupuesto.describe(include="all")
alto_presupuesto = dataset_limpio_rbf[(dataset_limpio_rbf['budget']>dataset_limpio_rbf['budget'].quantile(.50)) & (dataset_limpio_rbf['budget']<=dataset_limpio_rbf['budget'].quantile(.75))]
a_pre=alto_presupuesto.describe(include="all")
muy_alto_presupuesto = dataset_limpio_rbf[(dataset_limpio_rbf['budget']>dataset_limpio_rbf['budget'].quantile(.75))]
m_a_pre=muy_alto_presupuesto.describe(include="all")

#Generamos visualizaciones que nos indiquen por genero y rangos de presupuesto cuales son los que tienen mejor ratio beneficios

sns.catplot(y="genre",  kind="count", data=bajo_presupuesto)
plt.title('Número peliculas bajo presupuesto', fontsize = 15)
sns.catplot(x="ratio_beneficio_coste", y="genre", kind="box", data=bajo_presupuesto)
plt.title('Ratio beneficio coste vs genero', fontsize = 15)
sns.catplot(y="genre",  kind="count", data=medio_presupuesto)
plt.title('Número peliculas medio presupuesto', fontsize = 15)
sns.catplot(x="ratio_beneficio_coste", y="genre", kind="box", data=medio_presupuesto)
plt.title('Ratio beneficio coste vs genero', fontsize = 15)
sns.catplot(y="genre",  kind="count", data=alto_presupuesto)
plt.title('Número peliculas alto presupuesto', fontsize = 15)
sns.catplot(x="ratio_beneficio_coste", y="genre", kind="box", data=alto_presupuesto)
plt.title('Ratio beneficio coste vs genero', fontsize = 15)
sns.catplot(y="genre",  kind="count", data=muy_alto_presupuesto)
plt.title('Número peliculas muy alto presupuesto', fontsize = 15)
sns.catplot(x="ratio_beneficio_coste", y="genre", kind="box", data=muy_alto_presupuesto)
plt.title('Ratio beneficio coste vs genero', fontsize = 15)

#Realizamos tambien una division del dataset por genero para relacionarlo posteriormente con el ratio de beneficios y el mes
Comedy = dataset_limpio_rbf[ (dataset_limpio_rbf['genre']=='Comedy') ]
Action = dataset_limpio_rbf[ (dataset_limpio_rbf['genre']=='Action') ]
Drama = dataset_limpio_rbf[ (dataset_limpio_rbf['genre']=='Drama') ]
Crime = dataset_limpio_rbf[ (dataset_limpio_rbf['genre']=='Crime') ]
Adventure = dataset_limpio_rbf[ (dataset_limpio_rbf['genre']=='Adventure') ]
Biography = dataset_limpio_rbf[ (dataset_limpio_rbf['genre']=='Biography') ]
Animation = dataset_limpio_rbf[ (dataset_limpio_rbf['genre']=='Animation') ]
Horror = dataset_limpio_rbf[ (dataset_limpio_rbf['genre']=='Horror') ]

#Generamos las visualizaciones correspondientes
#Comedy
sns.catplot(x="released", y="ratio_beneficio_coste", kind="box", data=Comedy)
plt.title('Mes extreno vs Ratio beneficio coste', fontsize = 15)
sns.catplot(x="released",  kind="count", data=Comedy)
plt.title('peliculas extrenadas por mes', fontsize = 15)

#Action
sns.catplot(x="released", y="ratio_beneficio_coste", kind="box", data=Action)
plt.title('Mes extreno vs Ratio beneficio coste', fontsize = 15)
sns.catplot(x="released",  kind="count", data=Action)
plt.title('peliculas extrenadas por mes', fontsize = 15)

#Drama
sns.catplot(x="released", y="ratio_beneficio_coste", kind="box", data=Drama)
plt.title('Mes extreno vs Ratio beneficio coste', fontsize = 15)
sns.catplot(x="released",  kind="count", data=Drama)
plt.title('peliculas extrenadas por mes', fontsize = 15)

#Crime
sns.catplot(x="released", y="ratio_beneficio_coste", kind="box", data=Crime)
plt.title('Mes extreno vs Ratio beneficio coste', fontsize = 15)
sns.catplot(x="released",  kind="count", data=Crime)
plt.title('peliculas extrenadas por mes', fontsize = 15)

#Adventure
sns.catplot(x="released", y="ratio_beneficio_coste", kind="box", data=Adventure)
plt.title('Mes extreno vs Ratio beneficio coste', fontsize = 15)
sns.catplot(x="released",  kind="count", data=Adventure)
plt.title('peliculas extrenadas por mes', fontsize = 15)

#Biography
sns.catplot(x="released", y="ratio_beneficio_coste", kind="box", data=Biography)
plt.title('Mes extreno vs Ratio beneficio coste', fontsize = 15)
sns.catplot(x="released",  kind="count", data=Biography)
plt.title('peliculas extrenadas por mes', fontsize = 15)

#Animation
sns.catplot(x="released", y="ratio_beneficio_coste", kind="box", data=Animation)
plt.title('Mes extreno vs Ratio beneficio coste', fontsize = 15)
sns.catplot(x="released",  kind="count", data=Animation)
plt.title('peliculas extrenadas por mes', fontsize = 15)

#Horror
sns.catplot(x="released", y="ratio_beneficio_coste", kind="box", data=Horror)
plt.title('Mes extreno vs Ratio beneficio coste', fontsize = 15)
sns.catplot(x="released",  kind="count", data=Horror)
plt.title('peliculas extrenadas por mes', fontsize = 15)