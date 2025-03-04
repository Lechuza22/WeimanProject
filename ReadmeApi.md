# Weiman - Análisis de Datos y Predicciones con Streamlit

## Descripción del Proyecto
Este proyecto es una aplicación desarrollada en **Streamlit** que permite realizar análisis de datos y predicciones sobre ventas, clientes e inventario de una empresa. Utiliza algoritmos de **Machine Learning**, como **K-Means** para segmentación de clientes y productos, y modelos de series temporales como **ARIMA** y **Prophet** para predicciones de ventas y compras.

## Tecnologías Utilizadas
- **Python**
- **Streamlit** (Interfaz interactiva)
- **Pandas** (Manejo de datos)
- **Matplotlib** (Visualización)
- **Scikit-learn** (Modelos de Machine Learning)
- **Statsmodels** (Modelado ARIMA)
- **Prophet** (Predicciones de series temporales)

## Funcionalidades Principales
### 1. **Segmentación de Clientes**

- Se utiliza **K-Means** para agrupar clientes según su canal de compra.
- Evaluación del modelo mediante **Inertia**, **Silhouette Score**, **Davies-Bouldin Index** y **Calinski-Harabasz Score**.

### 2. **Análisis de Ventas y Predicciones**

- Filtrado de ventas con unidades mayores a 0.
- Agrupación de productos según su categoría utilizando **K-Means**.
- Predicción de ventas por **categoría**, **provincia** y **canal** mediante **ARIMA**.
- Evaluación de modelos ARIMA con **MAE (Mean Absolute Error)**, **MSE (Mean Squared Error)** y **RMSE (Root Mean Squared Error)**.
- Predicción de ventas con opción de horizonte de **7, 15, 30 o 60 días**.
- Filtrado de datos en los gráficos para mostrar solo el año más reciente disponible.
- Predicción de compras por canal con **Prophet**.

### 3. **Evolución de Compras**
- Análisis de compras por canal y evolución temporal mediante gráficos.

### 4. **Recomendación de Productos**
- Sistema de recomendación basado en la segmentación de productos mediante **clusters** de K-Means.

### 5. **Disponibilidad de Inventario**
- Verificación de disponibilidad de productos en inventario según su categoría.

## Estructura del Proyecto

weiman_project/ │── app.py # Código principal en Streamlit │── archivos_transformados/ # Carpeta con datasets de ventas, clientes e inventario │── requirements.txt # Dependencias del proyecto │── README.md # Documentación


## Instalación y Ejecución
1. **Clonar el repositorio:**
   ```sh
   git clone https://github.com/usuario/weiman_project.git
   cd weiman_project
2. **Instalar dependencias:**
   ```sh
      pip install -r requirements.txt
3. **Ejecutar la aplicación:**
   ```sh
     streamlit run app.py

## Resultados y Aplicaciones

Esta aplicación permite a las empresas:

- Optimizar la gestión de inventario.
- Identificar patrones de compra de clientes.
- Predecir tendencias de ventas para una mejor toma de decisiones.
- Implementar estrategias de recomendación de productos más efectivas.
