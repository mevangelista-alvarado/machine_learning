# Regresion\_Lineal\_en\_Iris\_con\_Sklearn

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mevangelista-alvarado/machine\_learning/blob/main/Regresion\_Lineal\_en\_Iris\_con\_Sklearn.ipynb)

**Actualizado:** Octubre de 2023

**Autor:** Miguel Evangelista ([@mevagelista-alvarado](https://github.com/mevangelista-alvarado))

### Introducción

En este Google Colab mostramos una implementación de Regresión Lineal con el el _dataset de la flor de Iris_ precargado en el módulo de Python `sklearn`.

### Iris Data Set

El conjunto de datos de la flor de Iris es un conjunto de datos multivariado utilizado por el estadístico y biólogo británico Ronald Fisher en su artículo "El uso de mediciones múltiples en problemas taxonómicos como ejemplo de análisis discriminante lineal" de 1936

El conjunto de datos consta de 50 muestras de cada una de las tres especies de Iris: Iris setosa, Iris virginica e Iris versicolor. De cada muestra se midieron (en centímetros) cuatro características : el largo y el ancho de los sépalos y pétalos.

```python
from sklearn.datasets import load_iris
```

Guardamos el dataset de la plata del iris en en la variable `iris`

```python
iris = load_iris()
```

```python
# @title
iris
```

Notamos que la variable iris es un diccionario de Python. Por tal, motivo verificamos las llaves que contiene.

```python
print(iris.keys())
```

```
dict_keys(['data', 'target', 'frame', 'target_names', 'DESCR', 'feature_names', 'filename', 'data_module'])
```

Notamos la llave _DESCR_, esta llave es importante por que debe contener información del dataset de la flor de Iris.

```python
print(iris.DESCR)
```

Exploramos más valores de algunas llaves.

```python
iris.target_names
```

```python
iris.target
```

```python
iris.data
```

```python
iris.feature_names
```

### Creando un DataFrame

Importamos el módulo `pandas` y creamos un Data Frame.

```python
import pandas as pd

iris_df = pd.DataFrame(data=iris['data'], columns=iris['feature_names'])
```

Exploramos el Data Frame

```python
iris_df
```

Agregamos una nueva columna llamada _species_ al Data Frame

```python
iris_df['species'] = iris["target"]
iris_df
```

Para tener una mejor visualización de datos definimos la siguiente función.

```python
def converter(specie):
    if specie == 0:
        return 'setosa'
    elif specie == 1:
        return 'versicolor'
    else:
        return 'virginica'
```

Y la aplicamos a los valores de la columna _'species'_

```python
iris_df['species'] = iris_df['species'].apply(converter)
iris_df
```

Obtenemos un pequeña descripción estadística del Data Frame

```python
iris_df.describe()
```

Con ayuda del módulo `seaborn` ontenemos una pequeña descripción gráfica del Data Frame.

Este método de forma predeterminada creará una cuadrícula de ejes con las variables numéricas del Data Frame.

Los gráficos diagonales se tratan de manera diferente: se dibuja un gráfico de distribución.

```python
import seaborn as sns
sns.pairplot(iris_df, hue='species')
```

Para más información de como funciona el método `pairplot` puede consultar la documentación https://seaborn.pydata.org/generated/seaborn.pairplot.html

### Regresion Lineal

Para realizar la regresión lineal necesitamos eliminar del Data Frame la columna de _'species'_ que contiene variables de tipo string.

```python
iris_df.drop('species', axis= 1, inplace= True)
iris_df
```

Vamos a realizar regresión lineal múltiple, de la siguiente manera:

* Nuestra variable a predecir (variable dependiente) será la contenida en la columna _'sepal length (cm)'_, denotada por la letra $Y$.
* Nuestros predictores (variables independientes) serán el resto de las columanas del data frame, denotada por la letra $X$.

```python
X = iris_df.drop(labels='sepal length (cm)', axis= 1)
Y = iris_df['sepal length (cm)']
```

Dividimos el Data Frame en datos de prueba con $\frac{1}{3}$ del total y datos de entrenamiento con $\frac{2}{3}$ del total. Para esto utilizamos la función `train_test_split` de `sklearn`.

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state= 101)
```

Para mas información de como funciona el método `train_test_split` puede consultar la documentación https://scikit-learn.org/stable/modules/generated/sklearn.model\_selection.train\_test\_split.html

Definimos un módelo de regresión lineal vacío.

```python
from sklearn.linear_model import LinearRegression

LR = LinearRegression()
```

Entrenamos el modelo con los datos de entrenamiento.

```python
LR.fit(X_train, y_train)
```

Hacemos las predicciones del módelo con los datos de prueba.

```python
pred = LR.predict(X_test)
pred
```

Obtenemos los valores de los coefientes de la ecuación de regresion.

```python
LR.coef_
```

Obtenemos el coefiente independiente.

```python
LR.intercept_
```

Por lo tanto la ecuación de regresión es

```python
print(f"f(x1,x2,x3) = x1*{LR.coef_[0]} + x2*{LR.coef_[1]} + x3*{LR.coef_[2]} + {LR.intercept_}")
```

Por último, revisamos las métricas para saber si el módelo de regresión es válido.

```python
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# Los valores cercanos al cero indican un mejor ajuste
print('Error cuadrático medio :', mean_squared_error(y_test, pred))
# El coeficiente de determinacion: 1 es una prediccion perfecta, r > 0 relacion directa y r < 0 relacion inversa.
print("Coeficiente de determinación: %.2f" % r2_score(y_test, pred))
```

Por, último comaparamos una predicción con el valor esperado.

```python
row = iris_df.iloc[1:3]
row
```

```python
target = 4.9
x = [[3.0,	1.4,	0.2]]
_pred = LR.predict(x)
_pred
```

**Ejercicio:**

Realiza la siguiente regresión, ahora toma variable dependiente otra columna distinta _'sepal length (cm)'_ y como variable independiente el resto del Data Frame y realiza la regresión lineal correspondiente.
