Análisis de Riesgo Crediticio con Machine Learning

¡Bienvenido a este proyecto de Análisis de Riesgo Crediticio utilizando Machine Learning! Este repositorio contiene código y recursos para entrenar y evaluar un modelo de regresión logística que predice la probabilidad de incumplimiento de pago de un solicitante de crédito.

Tabla de Contenidos

Descripción

Datos

Requisitos

Estructura del Proyecto

Instrucciones de Ejecución

Uso del Modelo

Solución de Problemas Comunes

Notas Adicionales

Créditos

Licencia


Descripción

El objetivo de este proyecto es desarrollar un modelo que ayude a instituciones financieras a evaluar el riesgo crediticio de potenciales clientes. Utilizamos un conjunto de datos real para entrenar un modelo de regresión logística, que es adecuado para problemas de clasificación binaria como predecir si un solicitante incumplirá o no con el pago de un préstamo.

Datos

Utilizamos el conjunto de datos credit_risk_dataset.csv obtenido de Kaggle. Este conjunto de datos incluye información relevante sobre solicitantes de crédito, como:

Edad

Ingresos

Tipo de vivienda

Historial de empleo

Monto del préstamo

Tasa de interés

Historial crediticio


Requisitos

Antes de ejecutar el proyecto, asegúrate de tener instalado lo siguiente:

Python 3.x

Las siguientes bibliotecas de Python:

pip install pandas numpy scikit-learn matplotlib joblib


Estructura del Proyecto

credit_risk_model.ipynb: Notebook de Jupyter con el código completo del proyecto.

credit_risk_dataset.csv: Conjunto de datos utilizado para entrenar y probar el modelo.

credit_risk_model.joblib: Archivo generado que contiene el modelo entrenado y objetos auxiliares.

README.md: Este documento.


Instrucciones de Ejecución

Sigue estos pasos para ejecutar el proyecto:

1. Clona el repositorio o descarga los archivos a tu máquina local.

git clone https://github.com/tu_usuario/analisis-riesgo-crediticio.git


2. Navega al directorio del proyecto:

cd analisis-riesgo-crediticio


3. Instala las dependencias:

pip install pandas numpy scikit-learn matplotlib joblib


4. Asegúrate de que el conjunto de datos está en el directorio:

Verifica que credit_risk_dataset.csv se encuentre en el directorio raíz del proyecto.



5. Ejecuta el notebook:

Abre credit_risk_model.ipynb con Jupyter Notebook o Google Colab.

Ejecuta todas las celdas en orden para entrenar el modelo y generar las evaluaciones.




Uso del Modelo

Entrenamiento y Evaluación

El script realiza las siguientes operaciones:

Carga y Preprocesamiento de Datos:

df = load_data('credit_risk_dataset.csv')
features, target, feature_columns, imputer, scaler = preprocess_data(df)

División de Datos:

X_train, X_test, y_train, y_test = split_data(features, target)

Entrenamiento del Modelo:

model = train_model(X_train, y_train)

Guardado del Modelo y Objetos Auxiliares:

joblib.dump((model, feature_columns, imputer, scaler), 'credit_risk_model.joblib')

Evaluación del Modelo:

accuracy, precision, recall, f1 = evaluate_model(model, X_test, y_test)
plot_metrics([accuracy, precision, recall, f1])


Evaluación de Nuevos Solicitantes

Para evaluar un nuevo solicitante de crédito:

1. Define los datos del prospecto en forma de diccionario:

prospect = {
    'person_age': 30,
    'person_income': 60000,
    'person_home_ownership': 'RENT',
    'person_emp_length': 10,
    'loan_intent': 'PERSONAL',
    'loan_grade': 'B',
    'loan_amnt': 10000,
    'loan_int_rate': 10.0,
    'loan_percent_income': 0.15,
    'cb_person_default_on_file': 'N',
    'cb_person_cred_hist_length': 5
}


2. Evalúa el prospecto utilizando las funciones proporcionadas:

prediction, prediction_proba = evaluate_prospect(prospect)
print(f'Prediction: {prediction}')
print(f'Prediction Probability: {prediction_proba}')

prediction será 0 o 1, indicando si es probable que incumpla (1) o no (0).

prediction_proba proporciona la probabilidad asociada a cada clase.




Solución de Problemas Comunes

Error con las Columnas de Características

Si encuentras un error relacionado con las columnas al evaluar un nuevo prospecto, es probable que se deba a que las características del prospecto no coinciden con las que el modelo espera.

Solución:

Durante el entrenamiento, se guardan las columnas de las características en feature_columns.

Al preprocesar el prospecto, usamos estas columnas para reindexar el DataFrame del prospecto:

prospect_features = pd.get_dummies(prospect_df, drop_first=True)
prospect_features = prospect_features.reindex(columns=feature_columns, fill_value=0)


Esto asegura que el modelo reciba las mismas características en el mismo orden que durante el entrenamiento.

Notas Adicionales

Personalización del Modelo: Puedes experimentar con diferentes modelos o ajustar los hiperparámetros para mejorar el rendimiento.

Ampliación del Conjunto de Datos: Utilizar más datos o datos más recientes puede mejorar la precisión del modelo.

Implementación en Producción: El modelo guardado puede integrarse en aplicaciones web o sistemas internos para la evaluación automatizada del riesgo crediticio.

Validación Cruzada: Para obtener una estimación más robusta del rendimiento del modelo, considera utilizar técnicas de validación cruzada.


Créditos

Autor: Agustín

Conjunto de Datos: Obtenido de Kaggle.

Inspiración: Este proyecto es parte de un viaje de aprendizaje en el campo del Machine Learning y el análisis de riesgo crediticio.


Licencia

Este proyecto está bajo la Licencia MIT.


---

¡Gracias por explorar este proyecto! Si tienes preguntas o sugerencias, no dudes en abrir un issue o contactarme directamente.

