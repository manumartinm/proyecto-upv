import streamlit as st
from datetime import datetime, timedelta
from data_managers.synthetic_data import SyntheticDataGenerator
from data_managers.weather import WeatherDataManager
from data_managers.red_electrica import RedElectricaDataManager
from model.electricity import EnergyOptimizer

st.title("Simulador de Optimización Energética con Paneles Solares")

st.sidebar.header("Parámetros de Simulación")
tipo_hogar = st.sidebar.selectbox("Tipo de Hogar", ['familia', 'pareja_joven', 'jubilado'])
latitud = st.sidebar.number_input("Latitud", value=39.46975, step=0.0001)
longitud = st.sidebar.number_input("Longitud", value=-0.37739, step=0.0001)
num_solar_panels = st.sidebar.number_input("Número de Paneles Solares", min_value=0, value=10, step=1)
potencia_panel_w = st.sidebar.number_input("Potencia de Panel Solar (W)", min_value=0, value=300, step=1)
bateria_max_kwh = st.sidebar.number_input("Capacidad Máxima de Batería (kWh)", min_value=0.0, value=10.0, step=0.5)
precio_compra_kw = st.sidebar.number_input("Precio de Compra de Electricidad (€/kWh)", min_value=0.0, value=0.4, step=0.01)
eficiencia_bateria = st.sidebar.slider("Eficiencia de Batería", min_value=0.0, max_value=1.0, value=0.95, step=0.01)
num_historical_hours = st.sidebar.number_input("Horas de Datos Históricos Sintéticos", min_value=100, value=24*30*6, step=24) # Ej: 6 meses
num_future_hours = st.sidebar.number_input("Horas a Predecir y Optimizar", min_value=24, value=48, step=12) # Ej: 2 días

if st.sidebar.button("Ejecutar Simulación"):
    st.header("Resultados de la Simulación")

    weather_data_manager = WeatherDataManager()

    weather_df = weather_data_manager.obtener_datos_clima(
        latitud=latitud,  # Latitud de Valencia
        longitud=longitud,  # Longitud de Valencia
        fecha_inicio=datetime.now() - timedelta(hours=num_historical_hours),
        fecha_fin=datetime.now() + timedelta(hours=num_future_hours),
    )

    # 1. Generar datos históricos sintéticos
    synthetic_data_generator = SyntheticDataGenerator(
        tipo_hogar=tipo_hogar,
        num_paneles=num_solar_panels,
        inicio=datetime.now() - timedelta(hours=num_historical_hours),
        dias=num_historical_hours // 24,
        meteo_df=weather_df,
        potencia_panel_w=potencia_panel_w,
    )

    st.info(f"Generando datos históricos sintéticos para {num_historical_hours} horas...")
    historical_data = synthetic_data_generator.generate_synthetic_data(num_hours=num_historical_hours, num_solar_panels=num_solar_panels)

    st.info(f"Obteniendo precios de electricidad de la API para las próximas {num_future_hours} horas...")
    price_fetcher = RedElectricaDataManager()
    # Intentar obtener precios desde la hora actual hasta future_hours en el futuro
    start_date_api = datetime.now().strftime('%Y-%m-%d')
    end_date_api = (datetime.now() + timedelta(hours=num_future_hours)).strftime('%Y-%m-%d')
    future_price_data = price_fetcher.obtener_precios_periodo(start_date_api, end_date_api)

    if future_price_data.empty:
        st.warning("No se pudieron obtener precios de la API. Se usarán precios sintéticos o históricos si están disponibles.")
        future_price_data = None # Asegurar que es None si la llamada falló o no devolvió datos


    # 3. Inicializar y cargar datos en el optimizador
    optimizer = EnergyOptimizer()
    # Pasamos los datos históricos sintéticos y, opcionalmente, los precios de la API para el futuro
    # El método load_and_preprocess_data dividirá historical_data en entrenamiento y futuro
    # e integrará future_price_data en la parte futura de historical_data si se proporciona.
    optimizer.load_and_preprocess_data(historical_data, future_hours=num_future_hours, price_dataframe=future_price_data)

    # Verificar si hay datos para continuar
    if optimizer.df is not None and not optimizer.df.empty:
        # 4. Entrenar modelos
        optimizer.train_models()

        # 5. Evaluar modelos (opcional)
        # optimizer.evaluate_models()

        # 6. Realizar predicciones futuras
        optimizer.predict_future()

        # 7. Ejecutar optimización
        optimizer.optimize_energy(bateria_max_kwh=bateria_max_kwh, precio_compra_kw=precio_compra_kw, eficiencia_bateria=eficiencia_bateria)

        # 8. Analizar y mostrar resultados
        optimizer.analyze_results(precio_compra_kw=precio_compra_kw)

    else:
        st.error("No se pudieron cargar o preprocesar los datos. La simulación no puede continuar.")

