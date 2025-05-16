import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import streamlit as st

class RedElectricaDataManager:
    def __init__(self):
        self.df_precios = None
        self.fecha_min = None
        self.fecha_max = None
    
    @staticmethod
    def obtener_precios_periodo(fecha_inicio, fecha_fin):
        url = (
            f"https://apidatos.ree.es/es/datos/mercados/precios-mercados-tiempo-real"
            f"?start_date={fecha_inicio}T00:00&end_date={fecha_fin}T23:59&time_trunc=hour"
        )
        headers = {"User-Agent": "Mozilla/5.0"}

        try:
            r = requests.get(url, headers=headers)
            r.raise_for_status()
            data = r.json()
            venta = data['included'][0]['attributes']['values']
            return pd.DataFrame(venta)
        except Exception as e:
            st.error(f"Error en fechas {fecha_inicio} a {fecha_fin}: {e}")
            return pd.DataFrame()
    
    def cargar_datos_completos(self, fecha_inicio, fecha_fin, delta_dias=7):
        fecha_base = datetime.strptime(fecha_inicio, '%Y-%m-%d')
        fecha_final = datetime.strptime(fecha_fin, '%Y-%m-%d')
        delta = timedelta(days=delta_dias)
        
        dfs = []
        
        with st.spinner(f'Cargando datos de precios de electricidad para el año {fecha_inicio}...'):
            while fecha_base < fecha_final:
                inicio = fecha_base.strftime('%Y-%m-%d')
                fin = (fecha_base + delta - timedelta(days=1)).strftime('%Y-%m-%d')
                df = self.obtener_precios_periodo(inicio, fin)
                if not df.empty:
                    dfs.append(df)
                fecha_base += delta
        
        if not dfs:
            st.error(f"No se pudieron obtener datos para el año {fecha_inicio}.")
            return pd.DataFrame()
            
        # Unimos todos los resultados
        df_completo = pd.concat(dfs, ignore_index=True)
        print(df_completo.head())
        # Procesamiento del dataframe
        df_completo['datetime'] = pd.to_datetime(df_completo['datetime'], errors='coerce', utc=True)
        df_completo['value'] = df_completo['value'] / 1000  # Convertir de Wh a kWh
        
        # Convertir valor a céntimos de euro y redondear
        df_completo['valor_centimos'] = df_completo['value'] * 100
        df_completo['valor_redondeado'] = df_completo['valor_centimos'].round(2)
        
        # Guardar el dataframe procesado
        self.df_precios = df_completo
        self.fecha_min = df_completo['datetime'].min()
        self.fecha_max = df_completo['datetime'].max()
        
        return df_completo
    
    def obtener_estadisticas(self):
        if self.df_precios is None or self.df_precios.empty:
            return {}
        
        stats = {
            'precio_medio': self.df_precios['valor_centimos'].mean().round(2),
            'precio_max': self.df_precios['valor_centimos'].max().round(2),
            'precio_min': self.df_precios['valor_centimos'].min().round(2),
            'fecha_precio_max': self.df_precios.loc[self.df_precios['valor_centimos'].idxmax(), 'datetime'],
            'fecha_precio_min': self.df_precios.loc[self.df_precios['valor_centimos'].idxmin(), 'datetime'],
            'desviacion_estandar': self.df_precios['valor_centimos'].std().round(2)
        }
        
        return stats
    
    def obtener_precios_por_periodo(self, tipo_periodo='mes'):
        if self.df_precios is None or self.df_precios.empty:
            return pd.DataFrame()
        
        if tipo_periodo == 'mes':
            return self.df_precios.groupby('mes')['valor_centimos'].agg(['mean', 'min', 'max']).round(2).reset_index()
        elif tipo_periodo == 'dia_semana':
            return self.df_precios.groupby('dia_semana')['valor_centimos'].agg(['mean', 'min', 'max']).round(2).reset_index()
        elif tipo_periodo == 'hora':
            return self.df_precios.groupby('hora')['valor_centimos'].agg(['mean', 'min', 'max']).round(2).reset_index()
        else:
            return pd.DataFrame()
    
    def filtrar_por_fechas(self, fecha_inicio, fecha_fin):
        if self.df_precios is None or self.df_precios.empty:
            return pd.DataFrame()
        
        mask = (self.df_precios['datetime'] >= fecha_inicio) & (self.df_precios['datetime'] <= fecha_fin)
        return self.df_precios[mask]