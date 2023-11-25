# Regresion\_Lineal\_simple\_ejemplo\_de\_juguete\_con\_Sklearn

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mevangelista-alvarado/machine\_learning/blob/main/Regresion\_Lineal\_simple\_ejemplo\_de\_juguete\_con\_Sklearn.ipynb)

**Actualizado:** Octubre de 2023

**Autor:** Miguel Evangelista ([@mevagelista-alvarado](https://github.com/mevangelista-alvarado))

### Introducción

En este Google Colab mostramos una implementación de Regresión Lineal de una variable con un ejemplo simple con el módulo de Python `sklearn`.

### Datos

Considere los siguiente datos. Tenemos calificaciones de las materias matemáticas y fisica de algunos alumnos.

```python
import pandas as pd

datos = {
    "mate": [2,3,4,4,5,6,6,7,7,8,10,10],
    "fisica": [1,3,2,4,4,4,6,4,6,7,9,10]
}

df = pd.DataFrame(data=datos)
df
```

|    | mate | fisica |
| -- | ---- | ------ |
| 0  | 2    | 1      |
| 1  | 3    | 3      |
| 2  | 4    | 2      |
| 3  | 4    | 4      |
| 4  | 5    | 4      |
| 5  | 6    | 4      |
| 6  | 6    | 6      |
| 7  | 7    | 4      |
| 8  | 7    | 6      |
| 9  | 8    | 7      |
| 10 | 10   | 9      |
| 11 | 10   | 10     |

Con ayuda del módulo `seaborn` ontenemos una pequeña descripción gráfica del Data Frame.

```python
import seaborn as sns
sns.pairplot(df)
```

Vamos a realizar regresión lineal, de la siguiente manera:

* Nuestra variable a predecir (variable dependiente) será la contenida en la columna _'fisica'_, denotada por la letra $Y$.
* Nuestros predictores (variables independientes) serán el resto de las columanas del data frame, denotada por la letra $X$.

```python
X = df.drop(labels='fisica', axis= 1)
Y = df['fisica']
```

Definimos un módelo de regresión lineal vacío.

```python
from sklearn.linear_model import LinearRegression

regr = LinearRegression()
```

Entrenamos el modelo

```python
regr.fit(X,Y)
```

Hacemos las predicciones del módelo

```python
pred = regr.predict(X)
```

Obtenemos los valores de los coefientes de la ecuación de regresion.

```python
regr.coef_
```

Obtenemos el coefiente independiente.

```python
regr.intercept_
```

Por lo tanto la ecuación de regresión es

```python
print(f"f(x1) = x1*{regr.coef_[0]} + {regr.intercept_}")
```

Visualización de la recta de regresión

```python
import matplotlib.pyplot as plt
plt.scatter(X, Y, color="black")
plt.plot(X, pred, color="blue", linewidth=3)
```

Por último, revisamos las métricas para saber si el módelo de regresión es válido.

```python
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# Los valores cercanos al cero indican un mejor ajuste
print('Error cuadrático medio :', mean_squared_error(Y, pred))
# El coeficiente de determinacion: 1 es una prediccion perfecta, r > 0 relacion directa y r < 0 relacion inversa.
print("Coeficiente de determinación: %.2f" % r2_score(Y, pred))
```

**Ejercicio:**

Realiza la siguiente regresión, ahora toma variable dependiente la columna de _'Mate'_ y como variable independiente la columna _'Fisica'_ y realiza la regresión lineal correspondiente.

```python
# Escribe aquí tu código
```
