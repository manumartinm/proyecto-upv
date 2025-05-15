import openmeteo_requests
import requests_cache
import pandas as pd
from retry_requests import retry
import streamlit as st

class WeatherDataManager:
    """
    Clase para obtener, procesar y almacenar datos meteorológicos
    utilizando la API de Open-Meteo.
    """
    
    def __init__(self):
        """
        Inicializa la clase ClimaDatos configurando la API de Open-Meteo.
        """
        # Setup the Open-Meteo API client with cache and retry on error
        self.cache_session = requests_cache.CachedSession('.cache', expire_after=-1)
        self.retry_session = retry(self.cache_session, retries=5, backoff_factor=0.2)
        self.openmeteo = openmeteo_requests.Client(session=self.retry_session)
        self.df_clima = None
        
    def obtener_datos_clima(self, latitud=39.46975, longitud=-0.37739, 
                          fecha_inicio="2024-01-01", fecha_fin="2024-12-31"):        
        with st.spinner('Obteniendo datos meteorológicos...'):
            # Definir parámetros de la API
            url = "https://archive-api.open-meteo.com/v1/archive"
            params = {
                "latitude": latitud,
                "longitude": longitud,
                "start_date": fecha_inicio.strftime('%Y-%m-%d'),
                "end_date": fecha_fin.strftime('%Y-%m-%d'),
                "hourly": [
                    "temperature_2m", 
                    "rain", 
                    "relative_humidity_2m", 
                    "apparent_temperature", 
                    "cloud_cover", 
                    "is_day", 
                    "sunshine_duration"
                ],
                "timezone": "auto"
            }
            
            try:
                # Realizar la petición a la API
                responses = self.openmeteo.weather_api(url, params=params)
                response = responses[0]
                
                # Información de respuesta
                info = {
                    "coordenadas": f"{response.Latitude()}°N {response.Longitude()}°E",
                    "elevacion": f"{response.Elevation()} m asl",
                    "zona_horaria": f"{response.Timezone()} {response.TimezoneAbbreviation()}",
                    "utc_offset": f"{response.UtcOffsetSeconds()} s"
                }
                
                # Procesar datos por hora
                hourly = response.Hourly()
                hourly_temperature_2m = hourly.Variables(0).ValuesAsNumpy()
                hourly_rain = hourly.Variables(1).ValuesAsNumpy()
                hourly_relative_humidity_2m = hourly.Variables(2).ValuesAsNumpy()
                hourly_apparent_temperature = hourly.Variables(3).ValuesAsNumpy()
                hourly_cloud_cover = hourly.Variables(4).ValuesAsNumpy()
                hourly_is_day = hourly.Variables(5).ValuesAsNumpy()
                hourly_sunshine_duration = hourly.Variables(6).ValuesAsNumpy()

                print("Datos meteorológicos obtenidos con éxito.")

                
                # Crear DataFrame
                hourly_data = {
                    "date": pd.date_range(
                        start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
                        end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
                        freq=pd.Timedelta(seconds=hourly.Interval()),
                        inclusive="left"
                    )
                }
                
                hourly_data["temperature_2m"] = hourly_temperature_2m
                hourly_data["rain"] = hourly_rain
                hourly_data["relative_humidity_2m"] = hourly_relative_humidity_2m
                hourly_data["apparent_temperature"] = hourly_apparent_temperature
                hourly_data["cloud_cover"] = hourly_cloud_cover
                hourly_data["is_day"] = hourly_is_day
                hourly_data["sunshine_duration"] = hourly_sunshine_duration

                print("Datos meteorológicos procesados con éxito.")
                
                # Crear DataFrame final
                df = pd.DataFrame(data=hourly_data)
                
                # Añadir columnas adicionales para análisis
                df['fecha'] = df['date'].dt.date
                df['hora'] = df['date'].dt.hour
                df['dia_semana'] = df['date'].dt.dayofweek
                df['mes'] = df['date'].dt.month
                
                # Convertir timezone a local
                df['date'] = df['date'].dt.tz_convert('Europe/Madrid')
                
                # Almacenar DataFrame en la instancia
                self.df_clima = df
                
                return df
                
            except Exception as e:
                st.error(f"Error al obtener datos meteorológicos: {str(e)}")
                return pd.DataFrame()
    
    def unir_con_precios(self, df_precios):
        if self.df_clima is None:
            st.warning("No hay datos de clima disponibles para unir con precios.")
            return df_precios
            
        if df_precios is None or df_precios.empty:
            st.warning("No hay datos de precios disponibles para unir con clima.")
            return self.df_clima
            
        # Convertir zona horaria de los precios si es necesario
        if df_precios['datetime'].dt.tz is None:
            df_precios_tz = df_precios.copy()
            df_precios_tz['datetime'] = pd.to_datetime(df_precios_tz['datetime']).dt.tz_localize('Europe/Madrid')
        else:
            df_precios_tz = df_precios
            
        # Unir dataframes por fecha y hora
        df_combinado = pd.merge(
            df_precios_tz,
            self.df_clima,
            left_on='datetime',
            right_on='date',
            how='inner'
        )
        
        return df_combinado
    
    def analizar_correlacion_precios(self, df_combinado):
        if df_combinado is None or df_combinado.empty:
            return pd.DataFrame()
            
        # Variables meteorológicas a correlacionar con el precio
        variables_meteo = [
            'temperature_2m', 
            'rain', 
            'relative_humidity_2m', 
            'apparent_temperature',
            'cloud_cover', 
            'is_day',
            'sunshine_duration'
        ]
        
        # Calcular correlaciones
        correlaciones = {}
        for var in variables_meteo:
            correlacion = df_combinado['valor_centimos'].corr(df_combinado[var])
            correlaciones[var] = round(correlacion, 3)
            
        # Convertir a DataFrame para mejor visualización
        df_correlaciones = pd.DataFrame(correlaciones.items(), columns=['Variable', 'Correlación'])
        df_correlaciones = df_correlaciones.sort_values('Correlación', ascending=False)
        
        return df_correlaciones
    
    def obtener_estadisticas_clima(self):
        if self.df_clima is None or self.df_clima.empty:
            return {}
        
        stats = {
            'temperatura_media': round(self.df_clima['temperature_2m'].mean(), 1),
            'temperatura_max': round(self.df_clima['temperature_2m'].max(), 1),
            'temperatura_min': round(self.df_clima['temperature_2m'].min(), 1),
            'humedad_media': round(self.df_clima['relative_humidity_2m'].mean(), 1),
            'lluvia_total': round(self.df_clima['rain'].sum(), 1),
            'dias_lluvia': len(self.df_clima[self.df_clima['rain'] > 0.1]['fecha'].unique()),
            'horas_sol_media': round(self.df_clima['sunshine_duration'].mean() / 60, 1)  # convertir de segundos a minutos
        }
        
        return stats
    
    def filtrar_por_condiciones(self, temperatura_min=None, temperatura_max=None, 
                              lluvia=None, solo_dia=False):
        if self.df_clima is None or self.df_clima.empty:
            return pd.DataFrame()
            
        df_filtrado = self.df_clima.copy()
        
        if temperatura_min is not None:
            df_filtrado = df_filtrado[df_filtrado['temperature_2m'] >= temperatura_min]
            
        if temperatura_max is not None:
            df_filtrado = df_filtrado[df_filtrado['temperature_2m'] <= temperatura_max]
            
        if lluvia is not None:
            if lluvia:
                df_filtrado = df_filtrado[df_filtrado['rain'] > 0.1]  # Más de 0.1 mm se considera lluvia
            else:
                df_filtrado = df_filtrado[df_filtrado['rain'] <= 0.1]
                
        if solo_dia:
            df_filtrado = df_filtrado[df_filtrado['is_day'] == 1]
            
        return df_filtrado
