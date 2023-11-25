# Regresion\_Logistica\_en\_IrisDataset\_con\_Sklearn

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mevangelista-alvarado/machine\_learning/blob/main/Regresion\_Logistica\_en\_IrisDataset\_con\_Sklearn.ipynb)

**Actualizado:** Octubre de 2023

**Autor:** Miguel Evangelista ([@mevagelista-alvarado](https://github.com/mevangelista-alvarado))

### Introducción

En este Google Colab mostramos una implementación de Regresión Logística con el el _dataset de la flor de Iris_ precargado en el módulo de Python `sklearn`.

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

Notamos que la variable iris es un diccionario de Python. Por tal, motivo verificamos las llaves que contiene.

```python
print(iris.keys())
```

Notamos la llave _DESCR_, esta llave es importante por que debe contener información del dataset de la flor de Iris.

```python
print(iris.DESCR)
```

Exploramos más valores de algunas llaves.

```python
iris.target_names
```

```
array(['setosa', 'versicolor', 'virginica'], dtype='<U10')
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

### Regresion Logística

Obtenemos los valores de $X$ e $Y$

```python
X = iris_df.drop(labels='species', axis= 1)
Y = iris_df['species']
```

Dividimos el Data Frame en datos de prueba con $\frac{1}{3}$ del total y datos de entrenamiento con $\frac{2}{3}$ del total. Para esto utilizamos la función `train_test_split` de `sklearn`.

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state= 42)
```

Para mas información de como funciona el método `train_test_split` puede consultar la documentación https://scikit-learn.org/stable/modules/generated/sklearn.model\_selection.train\_test\_split.html

En este caso, el valor 42 en `random_state` garantiza que la división de datos sea consistente en cada ejecución.

Es importante destacar que el valor específico que se proporciona a `random_state` es arbitrario; lo que importa es que sea el mismo en cada ejecución si deseas resultados consistentes.

No obstante, si no proporcionas ningún valor a `random_state`, se utilizará una semilla aleatoria diferente en cada ejecución, lo que puede generar resultados ligeramente diferentes cada vez.

#### Preprocesamiento de los Datos

La estandarización es un paso común en el preprocesamiento de datos que tiene como objetivo transformar las características para que tengan una media de $0$ y una desviación estándar de $1$.

Esto es útil cuando trabajas con algoritmos de aprendizaje automático, como la regresión logística, que pueden ser sensibles a las escalas de las características.

```python
from sklearn.preprocessing import StandardScaler

# Preprocesamiento: estandarizar las características
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
```

#### Definimos el modelo de regresión logística

Crear un modelo de regresión logística

```python
from sklearn.linear_model import LogisticRegression

logistic_regression = LogisticRegression(solver='lbfgs', multi_class='auto', max_iter=1000)
```

EL parametro `solver='lbfgs'` especifica el algoritmo de optimización utilizado para ajustar los parámetros del modelo. 'lbfgs' es un algoritmo de optimización que se utiliza comúnmente en la regresión logística.

Existen otros solucionadores disponibles como 'liblinear', 'newton-cg', 'sag', entre otros.

La elección del solucionador puede depender de la naturaleza de tus datos y del tamaño del conjunto de datos.

Por ejemplo, LBFGS es una abreviatura de "Limited-memory Broyden-Fletcher-Goldfarb-Shanno". Es un algoritmo de optimización que se utiliza comúnmente en la regresión logística y otros problemas de optimización en el campo del aprendizaje automático y la optimización numérica.

El algoritmo LBFGS es una variante del método de optimización BFGS (Broyden-Fletcher-Goldfarb-Shanno), que se utiliza para encontrar los valores óptimos de los parámetros en un modelo, como la regresión logística.

La principal característica distintiva del LBFGS es que utiliza una memoria limitada para almacenar la información necesaria para calcular las direcciones de búsqueda.

#### Entrenamiento

Entrenamos el modelo con los datos de entrenamiento.

```python
logistic_regression.fit(X_train, y_train)
```

#### Predicciones

Hacemos las predicciones del módelo con los datos de prueba.

```python
y_pred = logistic_regression.predict(X_test)
```

#### Evaluación del modelo

```python
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Evaluar el rendimiento del modelo
accuracy = accuracy_score(y_test, y_pred)
confusion = confusion_matrix(y_test, y_pred)
classification_report_str = classification_report(y_test, y_pred, target_names=iris.target_names)
```

Mostramos los resultados obtenidos

```python
print(f"Precisión del modelo: {accuracy:.2f}")
print("Matriz de confusión:")
print(confusion)
print("Informe de clasificación:")
print(classification_report_str)
```

Lo anterior, tiene la siguiente interpretación:

* **Precision:**\
  La precisión mide la proporción de predicciones correctas entre las instancias que el modelo ha clasificado como positivas.
* **Recall (Recuperación o Sensibilidad):**\
  El recall mide la proporción de instancias positivas reales que el modelo ha clasificado correctamente.
*   **F1-Score:**\
    El F1-Score es una métrica que combina la precisión y el recall en un solo valor. Es especialmente útil cuando deseas equilibrar la precisión y el recall.

    El F1-Score para cada clase se calcula como 2 \* (precision \* recall) / (precision + recall).
* **Support:**\
  El soporte indica el número de instancias de cada clase en el conjunto de prueba.
