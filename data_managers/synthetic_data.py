import pandas as pd
import numpy as np
from datetime import timedelta, datetime

class SyntheticDataGenerator:
    def __init__(self,
                 tipo_hogar='familia',
                 inicio='2024-01-01',
                 dias=366,
                 intervalo_minutos=60,
                 meteo_df=None,
                 num_paneles=8,
                 potencia_panel_w=390):
        if tipo_hogar not in ['familia', 'pareja_joven', 'jubilado']:
            raise ValueError("Tipo de hogar no reconocido. Debe ser 'familia', 'pareja_joven' o 'jubilado'.")

        if meteo_df is None or meteo_df.empty:
            print("Warning: No meteorological data provided (meteo_df is None or empty).")

        self.tipo_hogar = tipo_hogar
        self.inicio = inicio
        self.dias = dias
        self.intervalo_minutos = intervalo_minutos
        self.meteo_df = meteo_df
        self.num_paneles = num_paneles
        self.potencia_panel_w = potencia_panel_w
        self.potencia_total_kw = (num_paneles * potencia_panel_w) / 1000

        self._multiplicador_mes = {
            1: 1.10, 2: 1.05, 3: 1.00, 4: 0.97, 5: 0.95, 6: 0.98,
            7: 1.02, 8: 1.03, 9: 0.98, 10: 1.00, 11: 1.05, 12: 1.08
        }

        self._hora_amanecer_mes = {
            1: 8.25, 2: 7.45, 3: 7.3, 4: 7.0, 5: 6.3, 6: 6.15,
            7: 6.3, 8: 7.0, 9: 7.3, 10: 8.0, 11: 7.3, 12: 8.0
        }

        self._hora_atardecer_mes = {
            1: 18.0, 2: 18.5, 3: 19.5, 4: 20.0, 5: 21.0, 6: 21.5,
            7: 21.5, 8: 21.0, 9: 20.0, 10: 19.0, 11: 18.0, 12: 17.5
        }

        self._eficiencia = np.random.choice(['alta', 'media', 'baja'], p=[0.3, 0.5, 0.2])
        self._multiplicador_eficiencia = {'alta': 0.9, 'media': 1.0, 'baja': 1.1}
        self._limite_frio = np.random.uniform(9, 12)
        self._limite_calor = np.random.uniform(26, 29)
        self._setup_household_params()
        self._setup_vacations()

    def _setup_household_params(self):
        if self.tipo_hogar == 'familia':
            self._num_personas = np.random.choice([2, 3, 4, 5, 6], p=[0.1, 0.35, 0.35, 0.15, 0.05])
            self._vacaciones = np.random.rand() < (2 / 7)
        elif self.tipo_hogar == 'pareja_joven':
            self._num_personas = np.random.choice([1, 2, 3, 4], p=[0.1, 0.6, 0.2, 0.1])
            self._vacaciones = np.random.rand() < (4 / 10)
        elif self.tipo_hogar == 'jubilado':
            self._num_personas = np.random.choice([1, 2, 3], p=[0.6, 0.35, 0.05])
            self._jubilado_con_mas_luz = np.random.rand() < 0.6
            self._vacaciones = False

    def _setup_vacations(self):
        self._fechas_vacaciones = []
        if self._vacaciones:
            dias_vacaciones = np.random.randint(4, 16)
            fecha_inicio_dt = pd.to_datetime(self.inicio).tz_localize("UTC")
            inicio_vacaciones = fecha_inicio_dt + timedelta(days=np.random.randint(0, self.dias - dias_vacaciones))
            self._fechas_vacaciones = [inicio_vacaciones + timedelta(days=i) for i in range(dias_vacaciones)]

    def _get_base_consumption(self, hora, dia_semana):
        consumo_base = 0.0

        if self.tipo_hogar == 'familia':
            if dia_semana < 5: # Weekdays
                if 7 <= hora < 9: consumo_base = 0.65
                elif 14 <= hora < 16: consumo_base = 0.55
                elif 20 <= hora < 22: consumo_base = 1.0
                elif 1 <= hora < 7: consumo_base = 0.25
                else: consumo_base = 0.4
            else: # Weekends
                if 10 <= hora < 12: consumo_base = 0.7
                elif 19 <= hora < 21: consumo_base = 0.9
                elif 1 <= hora < 7: consumo_base = 0.25
                else: consumo_base = 0.4

        elif self.tipo_hogar == 'pareja_joven':
            if dia_semana < 5: # Weekdays
                if 8 <= hora < 9: consumo_base = 0.4
                elif 19 <= hora < 22: consumo_base = 0.7
                elif 0 <= hora < 7: consumo_base = 0.2
                else: consumo_base = 0.3
            else: # Weekends
                if 10 <= hora < 13: consumo_base = 0.6
                elif 18 <= hora < 21: consumo_base = 0.7
                elif 0 <= hora < 8: consumo_base = 0.2
                else: consumo_base = 0.3

        elif self.tipo_hogar == 'jubilado':
            if 7 <= hora < 9: consumo_base = 0.7
            elif 13 <= hora < 15: consumo_base = 0.65
            elif 20 <= hora < 22: consumo_base = 0.6
            elif 23 <= hora < 7: consumo_base = 0.2
            else: consumo_base = 0.35

        return consumo_base


    def _adjust_consumption_with_weather(self, consumo_base, ts, meteo):
        consumo_ajustado = consumo_base
        hora = ts.hour + ts.minute / 60
        mes = ts.month

        # Adjustments based on weather conditions
        if 'rain' in meteo and meteo['rain'] > 0:
            consumo_ajustado *= 1.04
        if 'apparent_temperature' in meteo and meteo['apparent_temperature'] < self._limite_frio:
            consumo_ajustado *= 1.07
        if 'apparent_temperature' in meteo and meteo['apparent_temperature'] > self._limite_calor:
            consumo_ajustado *= 1.07
        if 'relative_humidity_2m' in meteo and 'temperature_2m' in meteo and meteo['relative_humidity_2m'] > 80 and meteo['temperature_2m'] > 25:
            consumo_ajustado *= 1.02
        if 'sunshine_duration' in meteo and 'is_day' in meteo and meteo['sunshine_duration'] == 0 and meteo['is_day'] == 1:
            consumo_ajustado *= 1.03
        if 'cloud_cover' in meteo and 'is_day' in meteo and meteo['cloud_cover'] > 80 and meteo['is_day'] == 1 and \
           self._hora_amanecer_mes.get(mes, 6.0) < hora < self._hora_atardecer_mes.get(mes, 18.0):
            consumo_ajustado *= 1.02

        return consumo_ajustado
    
    def _calculate_generation(self, ts, hora, mes, meteo):
        if self._hora_amanecer_mes[mes] <= hora <= self._hora_atardecer_mes[mes]:
            rango = self._hora_atardecer_mes[mes] - self._hora_amanecer_mes[mes]
            x = (hora - self._hora_amanecer_mes[mes]) / rango
            perfil_solar = max(np.sin(np.pi * x), 0)
            sun_max = self.meteo_df['sunshine_duration'].max()

            if meteo['cloud_cover'] > 90 and meteo['sunshine_duration'] == 0:
                factor_clima = 0.05
            elif meteo['cloud_cover'] < 20 and meteo['sunshine_duration'] > 0:
                factor_clima = 1.0
            else:
                sunshine_frac = meteo['sunshine_duration'] / sun_max if sun_max > 0 else 0
                cloud_factor = 1 - meteo['cloud_cover'] / 100
                factor_clima = max(sunshine_frac * cloud_factor, 0.1)

            return round(self._multiplicador_eficiencia[self._eficiencia] * self.potencia_total_kw * perfil_solar * factor_clima, 3)
        return 0.0


    def generate_consumption(self):
        if self.meteo_df is None or self.meteo_df.empty:
            return None

        self.meteo_df['timestamp'] = self.meteo_df['date'].dt.tz_convert('UTC') if self.meteo_df['date'].dtype.tz else self.meteo_df['date'].dt.tz_localize('UTC')

        fecha_inicio_dt = pd.to_datetime(self.inicio).tz_localize("UTC")
        pasos = int((60 / self.intervalo_minutos) * 24 * self.dias)
        timestamps = [fecha_inicio_dt + timedelta(minutes=i * self.intervalo_minutos) for i in range(pasos)]

        consumo_data = []

        for ts in timestamps:
            esta_de_vacaciones = ts.date() in [d.date() for d in self._fechas_vacaciones]
            hora = ts.hour + ts.minute / 60
            mes = ts.month

            if esta_de_vacaciones:
                consumo = 0.05
            else:
                dia_semana = ts.weekday()
                temp_df = pd.DataFrame({'timestamp': [ts]})
                meteo_row_df = pd.merge_asof(temp_df, self.meteo_df, on='timestamp', direction='nearest')
                if meteo_row_df.empty or meteo_row_df.iloc[0]['timestamp'] != ts:
                    continue

                meteo = meteo_row_df.iloc[0]
                consumo_base = self._get_base_consumption(hora, dia_semana)
                consumo_ajustado = self._adjust_consumption_with_weather(consumo_base, ts, meteo)
                consumo_final = consumo_ajustado * self._multiplicador_mes.get(mes, 1.0)
                consumo = consumo_final * self._multiplicador_eficiencia.get(self._eficiencia, 1.0)
                consumo += np.random.normal(0, consumo * 0.05)
                consumo = max(0, consumo)

            meteo_dict = meteo.to_dict()

            consumo_data.append({
                'timestamp': ts,
                'dayweek': ts.day,
                'month': ts.month,
                'hour': ts.hour,
                'year': ts.year,
                'consumo_kwh': round(consumo, 3),
                'tipo_hogar': self.tipo_hogar,
                'num_personas': self._num_personas,
                'eficiencia_energetica': self._eficiencia,
                'generacion_kwh': self._calculate_generation(ts, hora, mes, meteo) if not esta_de_vacaciones else 0.0,
                'num_paneles': self.num_paneles,
                'potencia_panel_w': self.potencia_panel_w,
                'potencia_total_kw': round(self.potencia_total_kw, 2),
                **meteo_dict
            })
    
        df_consumo = pd.DataFrame(consumo_data)

        df_consumo['timestamp'] = pd.to_datetime(df_consumo['timestamp'], errors='coerce', utc=True)

        return df_consumo