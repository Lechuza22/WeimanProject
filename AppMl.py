import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import os
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score


# Configurar el logo y el favicon
st.set_page_config(page_title="Weiman", page_icon="archivos_transformados/WM.jpg")

# Definir la ruta de la carpeta
base_path = "archivos_transformados/"

# Cargar datasets
ventas_file = os.path.join(base_path, "ventas_Hoja1.csv")
inventario_file = os.path.join(base_path, "mov_inventario_Hoja1.csv")
clientes_file = os.path.join(base_path, "clientes_Hoja1.csv")

if os.path.exists(ventas_file):
    ventas_df = pd.read_csv(ventas_file)
else:
    st.error(f"Error: No se encontró el archivo {ventas_file}. Verifica la ruta o sube el archivo.")
    st.stop()

if os.path.exists(inventario_file):
    inventario_df = pd.read_csv(inventario_file)
else:
    st.error(f"Error: No se encontró el archivo {inventario_file}. Verifica la ruta o sube el archivo.")
    st.stop()

if os.path.exists(clientes_file):
    clientes_df = pd.read_csv(clientes_file)
else:
    st.error(f"Error: No se encontró el archivo {clientes_file}. Verifica la ruta o sube el archivo.")
    st.stop()

# Filtrar ventas con unidades mayores a 0
ventas_filtradas = ventas_df[ventas_df["unidades"] > 0]

# Aplicar K-Means para agrupación de productos con evaluación
ventas_agrupadas = ventas_filtradas.groupby("categoria")["unidades"].sum().reset_index()
scaler = StandardScaler()
ventas_agrupadas_scaled = scaler.fit_transform(ventas_agrupadas[["unidades"]])

num_clusters = 5
kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
ventas_agrupadas["cluster"] = kmeans.fit_predict(ventas_agrupadas_scaled)

# Evaluación del clustering
silhouette = silhouette_score(ventas_agrupadas_scaled, ventas_agrupadas["cluster"])
davies_bouldin = davies_bouldin_score(ventas_agrupadas_scaled, ventas_agrupadas["cluster"])
calinski_harabasz = calinski_harabasz_score(ventas_agrupadas_scaled, ventas_agrupadas["cluster"])

# Aplicar K-Means para agrupación de clientes por canal con evaluación
def segmentar_clientes():
    if "canal" not in clientes_df.columns:
        st.error("Error: La columna 'canal' no está presente en el dataset de clientes.")
        return pd.DataFrame()
    
    clientes_agrupados = clientes_df.groupby("canal").size().reset_index(name="cantidad_clientes")
    
    scaler = StandardScaler()
    clientes_scaled = scaler.fit_transform(clientes_agrupados[["cantidad_clientes"]])
    
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    clientes_agrupados["segmento"] = kmeans.fit_predict(clientes_scaled)
    
    # Evaluación del modelo
    inertia = kmeans.inertia_
    silhouette = silhouette_score(clientes_scaled, clientes_agrupados["segmento"])
    davies_bouldin = davies_bouldin_score(clientes_scaled, clientes_agrupados["segmento"])
    calinski_harabasz = calinski_harabasz_score(clientes_scaled, clientes_agrupados["segmento"])

    return clientes_agrupados

    
# Función para mostrar evolución de compras por canal
def evolucion_compras():
    compras_por_mes = clientes_df.groupby(["mes", "canal"]).agg({"importe": "sum"}).reset_index()
    compras_pivot = compras_por_mes.pivot(index="mes", columns="canal", values="importe").fillna(0)
    
    plt.figure(figsize=(10,5))
    for canal in compras_pivot.columns:
        plt.plot(compras_pivot.index, compras_pivot[canal], label=canal)
    plt.xlabel("Mes")
    plt.ylabel("Importe Total")
    plt.title("Evolución de Compras por Canal")
    plt.legend()
    st.pyplot(plt)

# Función para predecir compras por canal con Prophet
def predecir_compras():
    canal_seleccionado = st.selectbox("Selecciona un canal para predecir", clientes_df["canal"].unique())
    compras_por_mes = clientes_df[clientes_df["canal"] == canal_seleccionado].groupby("mes").agg({"importe": "sum"}).reset_index()
    compras_por_mes.columns = ["ds", "y"]
    
    modelo = Prophet()
    modelo.fit(compras_por_mes)
    futuro = modelo.make_future_dataframe(periods=12, freq='M')
    prediccion = modelo.predict(futuro)
    
    fig = modelo.plot(prediccion)
    st.write(f"Predicción de compras para el canal: {canal_seleccionado}")
    st.pyplot(fig)

# Función para recomendar productos dentro del mismo cluster
def recomendar_productos_cluster(categoria):
    cluster_id = ventas_agrupadas.loc[ventas_agrupadas["categoria"] == categoria, "cluster"].values
    if len(cluster_id) > 0:
        cluster_id = cluster_id[0]
        recomendaciones = ventas_agrupadas[ventas_agrupadas["cluster"] == cluster_id]["categoria"].tolist()
        if categoria in recomendaciones:
            recomendaciones.remove(categoria)  # Quitar la misma categoría seleccionada
        return recomendaciones[:5]  # Devolver hasta 5 recomendaciones
    return []

# Función para verificar disponibilidad de productos en inventario
def verificar_disponibilidad_categoria(categoria):
    productos_categoria = ventas_df[ventas_df["categoria"] == categoria]["codigo_art"].unique()
    inventario_filtrado = inventario_df[inventario_df["codigo_art"].isin(productos_categoria)]
    if not inventario_filtrado.empty:
        inventario_reciente = inventario_filtrado.sort_values(by="fecha", ascending=False).groupby("codigo_art").first().reset_index()
        stock_disponible = inventario_reciente[["codigo_art", "fecha", "egreso"]]
        stock_disponible = stock_disponible.merge(ventas_df[["codigo_art", "categoria"]].drop_duplicates(), on="codigo_art", how="left")
        stock_disponible = stock_disponible.rename(columns={"categoria": "Nombre del Artículo", "egreso": "Stock Disponible", "fecha": "Última Reposición"})
        return stock_disponible
    return None


# Función para predecir ventas con ARIMA y evaluar la predicción
def predecir_ventas(filtro_df, titulo):
    filtro_df.index = pd.to_datetime(filtro_df.index)
    filtro_df = filtro_df.resample('D').sum()  # Se asume granularidad diaria

    if len(filtro_df) > 30:  # Se necesita historial suficiente para predecir
        # Selección del horizonte de predicción
        periodo_prediccion = st.selectbox("Selecciona el horizonte de predicción:",
                                          [7, 15, 30, 60], index=2)

        # Filtrar solo el último año disponible en los datos
        ultimo_anio = filtro_df.index.max().year
        filtro_df = filtro_df[filtro_df.index.year == ultimo_anio]

        # Dividir datos en entrenamiento y prueba
        train_size = int(len(filtro_df) * 0.8)
        train, test = filtro_df.iloc[:train_size], filtro_df.iloc[train_size:]

        # Ajustar modelo ARIMA con datos de entrenamiento
        modelo = ARIMA(train, order=(5,1,0))
        modelo_fit = modelo.fit()

        # Realizar predicción sobre los datos de prueba
        predicciones = modelo_fit.forecast(steps=len(test))

        # Calcular métricas de evaluación
        mae = mean_absolute_error(test, predicciones)
        mse = mean_squared_error(test, predicciones)
        rmse = np.sqrt(mse)

        # Predicción futura
        future_pred = modelo_fit.forecast(steps=periodo_prediccion)

        # Graficar la predicción mostrando solo datos desde el último año
        plt.figure(figsize=(10, 5))
        plt.plot(filtro_df.index, filtro_df, label="Historial de Ventas")
        plt.plot(pd.date_range(filtro_df.index[-1], periods=periodo_prediccion+1, freq='D')[1:], 
                 future_pred, linestyle='dashed', color='red', label=f"Predicción ARIMA ({periodo_prediccion} días)")
        plt.legend()
        plt.xlabel("Fecha")
        plt.ylabel("Unidades Vendidas")
        plt.title(f"{titulo} - Predicción de {periodo_prediccion} días")
        st.pyplot(plt)

    else:
        st.write("No hay suficientes datos para predecir.")


# Interfaz de Streamlit
logo_path = os.path.join(base_path, "WM.jpg")
if os.path.exists(logo_path):
    st.image(logo_path, width=150)
st.title("Weimansolutions")

# Menú de opciones con botones en la barra lateral
st.sidebar.header("Menú")
menu = st.sidebar.radio("Selecciona una opción", ["Clientes", "Inventario", "Ventas", "Sistema de Recomendación de Productos"])

if menu == "Clientes":
    sub_menu = st.radio("Selecciona una sub-sección", ["Segmentación de Clientes", "Evolución de Compras por Canal", "Predicción de Compras por Canal"])
    
    if sub_menu == "Segmentación de Clientes":
        st.header("Segmentación de Clientes por Canal")
        clientes_segmentados = segmentar_clientes()
        
        if not clientes_segmentados.empty:
            st.write("Segmentación de clientes basada en el canal de compra:")
            st.dataframe(clientes_segmentados)
            
            plt.figure(figsize=(8,5))
            plt.bar(clientes_segmentados["canal"], clientes_segmentados["cantidad_clientes"], color="skyblue")
            plt.xlabel("Canal de Venta")
            plt.ylabel("Cantidad de Clientes")
            plt.title("Segmentación de Clientes por Canal")
            st.pyplot(plt)
        else:
            st.write("No hay datos suficientes para segmentar clientes.")
    
    elif sub_menu == "Evolución de Compras por Canal":
        st.header("Evolución de Compras por Canal")
        evolucion_compras()
    
    elif sub_menu == "Predicción de Compras por Canal":
        st.header("Predicción de Compras por Canal")
        predecir_compras()
        
if menu == "Inventario":
    st.header("Disponibilidad de Productos en Inventario")
    categoria_seleccionada = st.selectbox("Selecciona una categoría", ventas_df["categoria"].unique(), key="categoria_inventario")
    stock_disponible = verificar_disponibilidad_categoria(categoria_seleccionada)
    if stock_disponible is not None:
        st.write("Stock disponible en inventario:")
        st.dataframe(stock_disponible)
    else:
        st.write("No hay datos de inventario disponibles para esta categoría.")

if menu == "Ventas":
    sub_menu = st.radio("Selecciona una sub-sección", ["Predicción por Categoría", "Predicción por Provincia y Canal", "Proyección de Ventas por Canal"])
    
    if sub_menu == "Predicción por Categoría":
        categoria_seleccionada = st.selectbox("Selecciona una categoría", ventas_df["categoria"].unique(), key="categoria_prediccion")
        ventas_categoria = ventas_filtradas[ventas_filtradas["categoria"] == categoria_seleccionada].groupby("fecha")["unidades"].sum()
        predecir_ventas(ventas_categoria, f"Predicción de Ventas para {categoria_seleccionada}")
    
    elif sub_menu == "Predicción por Provincia y Canal":
        provincia_seleccionada = st.selectbox("Selecciona una provincia", ventas_df["provincia"].unique(), key="provincia_prediccion")
        canal_seleccionado = st.selectbox("Selecciona un canal", ["Minorista", "Online"], key="canal_prediccion")
        
        filtro_ventas = ventas_filtradas[(ventas_filtradas["provincia"] == provincia_seleccionada) & (ventas_filtradas["canal"] == canal_seleccionado)]
        ventas_filtro = filtro_ventas.groupby("fecha")["unidades"].sum()
        predecir_ventas(ventas_filtro, f"Predicción de Ventas en {provincia_seleccionada} para {canal_seleccionado}")
    
    elif sub_menu == "Proyección de Ventas por Canal":
        st.header("Proyección de Ventas por Canal para 2025")
        canales = ["Minorista", "Online"]
        ventas_canal = ventas_filtradas[ventas_filtradas["canal"].isin(canales)].groupby(["fecha", "canal"])["unidades"].sum().unstack()
        ventas_canal.index = pd.to_datetime(ventas_canal.index)
        ventas_canal = ventas_canal.resample('M').sum()
        
        proyeccion = ventas_canal * 1.05  # Aumento del 5% para 2025
        
        plt.figure(figsize=(10,5))
        for canal in canales:
            plt.plot(ventas_canal.index, ventas_canal[canal], label=f"{canal} - Historial")
            plt.plot(proyeccion.index, proyeccion[canal], linestyle='dashed', label=f"{canal} - Proyección 2025")
        plt.legend()
        plt.xlabel("Fecha")
        plt.ylabel("Unidades Vendidas")
        plt.title("Proyección de Ventas por Canal para 2025")
        st.pyplot(plt)
        
elif menu == "Sistema de Recomendación de Productos":
    st.header("Recomendación de Productos por Categoría")
    categoria_seleccionada = st.selectbox("Selecciona una categoría", ventas_df["categoria"].unique(), key="categoria_select")
    cluster_id = ventas_agrupadas.loc[ventas_agrupadas["categoria"] == categoria_seleccionada, "cluster"].values
    if len(cluster_id) > 0:
        cluster_id = cluster_id[0]
        recomendaciones = ventas_agrupadas[ventas_agrupadas["cluster"] == cluster_id]["categoria"].tolist()
        if categoria_seleccionada in recomendaciones:
            recomendaciones.remove(categoria_seleccionada)
        if recomendaciones:
            st.write("Categorías recomendadas en el mismo cluster:")
            for cat in recomendaciones:
                st.write(f"- {cat}")
        else:
            st.write("No hay suficientes datos para recomendar categorías relacionadas.")
