import pandas as pd
import numpy as np
import streamlit as st
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from pulp import LpProblem, LpMaximize, LpVariable, lpSum, LpStatus

class EnergyOptimizer:
    def __init__(self):
        """
        Inicializa la clase. No necesita file_path si los datos se pasan directamente.
        """
        self.df = None # DataFrame para datos de entrenamiento
        self.df_future = None # DataFrame para datos futuros (predicción y optimización)
        self.model_consumo = None
        self.model_generacion = None
        self.model_precio = None

        # Definición de las características para cada modelo
        self.features_consumo = ['hour', 'dayweek', 'month',
                                 'temperature_2m', 'rain', 'cloud_cover', 'sunshine_duration']
        self.features_generacion = ['hour', 'dayweek', 'month',
                                    'temperature_2m', 'rain', 'cloud_cover', 'sunshine_duration']
        self.features_precio = ['hour', 'dayweek', 'month',
                                'temperature_2m', 'rain', 'cloud_cover', 'sunshine_duration']

    def load_and_preprocess_data(self, historical_df: pd.DataFrame, future_hours: int = 48, price_dataframe: pd.DataFrame = None):
        try:
            # 1. Usar el DataFrame histórico proporcionado
            self.df = historical_df.copy()
            st.info(f"Datos históricos cargados. Filas: {len(self.df)}")

            # 2. Integrar datos de precios si se proporcionan
            if price_dataframe is not None and isinstance(price_dataframe, pd.DataFrame) and not price_dataframe.empty:
                st.info("Integrando datos de precios desde el DataFrame proporcionado...")
                # Renombrar columnas del DataFrame de precios para que coincidan con el formato esperado
                print(price_dataframe.head())
                price_df_renamed = price_dataframe.rename(columns={'datetime': 'timestamp', 'value': 'precio_electricidad_eur_kwh'})
                # Asegurar que la columna de precio es numérica
                price_df_renamed['precio_electricidad_eur_kwh'] = pd.to_numeric(price_df_renamed['precio_electricidad_eur_kwh'], errors='coerce')
                # Asegurar que la columna timestamp es datetime
                price_df_renamed['timestamp'] = pd.to_datetime(price_df_renamed['timestamp'])

                # Fusionar con el DataFrame principal. Usamos un merge 'left' para mantener todas las filas del df principal.
                original_cols = set(self.df.columns)

                print(price_df_renamed.head())

                self.df = pd.merge(self.df, price_df_renamed[['timestamp', 'precio_electricidad_eur_kwh']],
                                   on='timestamp', how='left', suffixes=('', '_new_price'))
                
                print(self.df.head())
                print(self.df.columns)

                if 'precio_electricidad_eur_kwh_new_price' in self.df.columns:
                     st.info("Sobrescribiendo columna de precio existente con los datos proporcionados.")
                     self.df['precio_electricidad_eur_kwh'] = self.df['precio_electricidad_eur_kwh_new_price']
                     self.df = self.df.drop(columns=['precio_electricidad_eur_kwh_new_price'])
                elif 'precio_electricidad_eur_kwh' not in original_cols and 'precio_electricidad_eur_kwh' in self.df.columns:
                     st.info("Columna de precio añadida desde el DataFrame proporcionado.")
                elif 'precio_electricidad_eur_kwh' in original_cols and 'precio_electricidad_eur_kwh' not in self.df.columns:
                     st.warning("Advertencia: La columna de precio original parece haber desaparecido después del merge.")


                st.info("Integración de precios completada.")
                if self.df['precio_electricidad_eur_kwh'].isnull().any():
                     st.warning("Advertencia: Se encontraron valores NaN en la columna de precio después de la integración.")


            elif 'precio_electricidad_eur_kwh' not in self.df.columns:
                 st.warning("Advertencia: No se proporcionó un DataFrame de precios y la columna 'precio_electricidad_eur_kwh' no está en los datos históricos.")
                 st.warning("El modelo de precio y la optimización podrían no funcionar correctamente.")
                 self.df['precio_electricidad_eur_kwh'] = np.nan # Asegurar que la columna existe aunque esté vacía


            # 3. Separar datos para predicción futura
            if len(self.df) > future_hours:
                start_time_future = self.df['timestamp'].max() - pd.Timedelta(hours=future_hours - 1)
                self.df_future = self.df[self.df['timestamp'] >= start_time_future].copy()
                self.df = self.df[self.df['timestamp'] < start_time_future].copy()
                st.info(f"Separados {len(self.df_future)} horas para predicción futura.")
                st.info(f"Usando {len(self.df)} horas para entrenamiento.")
            else:
                 st.warning(f"El DataFrame tiene menos de {future_hours} horas. Usando todo para entrenamiento.")
                 self.df_future = pd.DataFrame() # DataFrame vacío para futuro si no hay suficientes datos
                 # En este caso, self.df ya contiene todos los datos cargados.


            # 4. Crear features de tiempo en los DataFrames
            if not self.df.empty:
                self.df['hour'] = self.df['timestamp'].dt.hour
                self.df['dayweek'] = self.df['timestamp'].dt.weekday
                self.df['month'] = self.df['timestamp'].dt.month
                st.info("Features de tiempo creadas en datos de entrenamiento.")

            if not self.df_future.empty:
                 self.df_future['hour'] = self.df_future['timestamp'].dt.hour
                 self.df_future['dayweek'] = self.df_future['timestamp'].dt.weekday
                 self.df_future['month'] = self.df_future['timestamp'].dt.month
                 st.info("Features de tiempo creadas en datos futuros.")

        except Exception as e:
            st.error(f"Error durante la carga o preprocesamiento de datos: {e}")
            self.df = pd.DataFrame()
            self.df_future = pd.DataFrame()


    def train_models(self):
        """
        Entrena los modelos de Random Forest para consumo, generación y precio.
        """
        if self.df is None or self.df.empty:
            st.warning("No hay datos disponibles para entrenar los modelos.")
            return

        st.info("Entrenando modelos...")

        print(self.df.head())

        # Modelo de Consumo
        if 'consumo_kwh' in self.df.columns and all(f in self.df.columns for f in self.features_consumo):
            X_c = self.df[self.features_consumo].dropna() # Eliminar NaNs en features para entrenamiento
            y_c = self.df.loc[X_c.index, 'consumo_kwh'].dropna() # Asegurar que y coincide con X y no tiene NaNs
            # Asegurarse de que los índices coinciden después de dropna
            common_index_c = X_c.index.intersection(y_c.index)
            X_c = X_c.loc[common_index_c]
            y_c = y_c.loc[common_index_c]

            if not X_c.empty:
                self.model_consumo = RandomForestRegressor(n_estimators=100, random_state=16)
                self.model_consumo.fit(X_c, y_c)
                st.success("Modelo de consumo entrenado.")
            else:
                st.warning("No hay datos limpios para entrenar el modelo de consumo.")
                self.model_consumo = None
        else:
            st.warning("Columnas necesarias para el modelo de consumo no encontradas.")
            self.model_consumo = None

        # Modelo de Generación
        if 'generacion_kwh' in self.df.columns and all(f in self.df.columns for f in self.features_generacion):
            X_g = self.df[self.features_generacion].dropna() # Eliminar NaNs en features para entrenamiento
            y_g = self.df.loc[X_g.index, 'generacion_kwh'].dropna() # Asegurar que y coincide con X y no tiene NaNs
            # Asegurarse de que los índices coinciden después de dropna
            common_index_g = X_g.index.intersection(y_g.index)
            X_g = X_g.loc[common_index_g]
            y_g = y_g.loc[common_index_g]

            if not X_g.empty:
                self.model_generacion = RandomForestRegressor(n_estimators=100, random_state=42)
                self.model_generacion.fit(X_g, y_g)
                st.success("Modelo de generación entrenado.")
            else:
                st.warning("No hay datos limpios para entrenar el modelo de generación.")
                self.model_generacion = None
        else:
            st.warning("Columnas necesarias para el modelo de generación no encontradas.")
            self.model_generacion = None

        # Modelo de Precio
        if 'precio_electricidad_eur_kwh' in self.df.columns and all(f in self.df.columns for f in self.features_precio):
            X_p = self.df[self.features_precio].dropna() # Eliminar NaNs en features para entrenamiento
            y_p = self.df.loc[X_p.index, 'precio_electricidad_eur_kwh'].dropna() # Asegurar que y coincide con X y no tiene NaNs
            # Asegurarse de que los índices coinciden después de dropna
            common_index_p = X_p.index.intersection(y_p.index)
            X_p = X_p.loc[common_index_p]
            y_p = y_p.loc[common_index_p]

            if not X_p.empty:
                self.model_precio = RandomForestRegressor(n_estimators=100, random_state=16)
                self.model_precio.fit(X_p, y_p)
                st.success("Modelo de precio entrenado.")
            else:
                 st.warning("No hay datos de precio válidos (no NaN) para entrenar el modelo de precio.")
                 self.model_precio = None

        else:
            st.warning("Columnas necesarias para el modelo de precio no encontradas.")
            self.model_precio = None

        st.info("Entrenamiento de modelos completado.")


    def evaluate_models(self, n_splits: int = 5):
        """
        Realiza validación cruzada temporal para evaluar los modelos entrenados.

        Args:
            n_splits: Número de divisiones para TimeSeriesSplit.

        Returns:
            Un DataFrame con las métricas de evaluación para cada modelo.
        """
        if self.df is None or self.df.empty:
            st.warning("No hay datos disponibles para evaluar los modelos.")
            return pd.DataFrame()

        st.info("Evaluando modelos con validación cruzada temporal...")

        metrics_data = []

        # Evaluación del modelo de Consumo
        if self.model_consumo and 'consumo_kwh' in self.df.columns and 'precio_electricidad_eur_kwh' in self.df.columns:
             X_c = self.df[self.features_consumo]
             y_c = self.df['consumo_kwh']
             price_series_c = self.df['precio_electricidad_eur_kwh']
             metrics_c = self._evaluate_single_model(self.model_consumo, X_c, y_c, price_series_c, n_splits)
             metrics_data.append({'Model': 'Consumption', **metrics_c})
             st.info("Evaluación del modelo de consumo completada.")
        else:
             st.warning("Modelo de consumo no disponible o faltan columnas para evaluación.")


        # Evaluación del modelo de Generación
        if self.model_generacion and 'generacion_kwh' in self.df.columns and 'precio_electricidad_eur_kwh' in self.df.columns:
             X_g = self.df[self.features_generacion]
             y_g = self.df['generacion_kwh']
             price_series_g = self.df['precio_electricidad_eur_kwh']
             metrics_g = self._evaluate_single_model(self.model_generacion, X_g, y_g, price_series_g, n_splits)
             metrics_data.append({'Model': 'Generation', **metrics_g})
             st.info("Evaluación del modelo de generación completada.")
        else:
             st.warning("Modelo de generación no disponible o faltan columnas para evaluación.")

        # Evaluación del modelo de Precio
        if self.model_precio and 'precio_electricidad_eur_kwh' in self.df.columns:
             df_precio_eval = self.df.dropna(subset=['precio_electricidad_eur_kwh']).copy()
             if not df_precio_eval.empty:
                 X_p = df_precio_eval[self.features_precio]
                 y_p = df_precio_eval['precio_electricidad_eur_kwh']
                 price_series_p = df_precio_eval['precio_electricidad_eur_kwh']
                 metrics_p = self._evaluate_single_model(self.model_precio, X_p, y_p, price_series_p, n_splits)
                 metrics_data.append({'Model': 'Price', **metrics_p})
                 st.info("Evaluación del modelo de precio completada.")
             else:
                 st.warning("No hay datos de precio válidos (no NaN) para evaluar el modelo de precio.")
        else:
            st.warning("Modelo de precio no disponible o faltan columnas para evaluación.")


        if metrics_data:
            df_eval = pd.DataFrame(metrics_data)
            st.subheader("Resultados de la Evaluación del Modelo")
            st.dataframe(df_eval)
            return df_eval
        else:
            st.warning("No se pudieron evaluar los modelos.")
            return pd.DataFrame()


    def _evaluate_single_model(self, model, X, y, price, n_splits=5):
        """
        Realiza validación cruzada temporal y devuelve media y desviación
        de MSE, R2, MAE y ECM para un solo modelo. (Método interno)
        """
        tscv = TimeSeriesSplit(n_splits=n_splits)
        metrics = {'mse':[], 'r2':[], 'mae':[], 'ecm':[]}

        calculate_ecm = not price.empty and len(price) == len(y) and not price.isnull().any()

        for train_idx, val_idx in tscv.split(X):
            X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

            X_tr_clean = X_tr.dropna(subset=self.features_consumo + ['hour', 'dayweek', 'month'])
            y_tr_clean = y_tr.loc[X_tr_clean.index].dropna()

            # Alinear X_tr_clean y y_tr_clean por índice
            common_index_eval = X_tr_clean.index.intersection(y_tr_clean.index)
            X_tr_clean = X_tr_clean.loc[common_index_eval]
            y_tr_clean = y_tr_clean.loc[common_index_eval]


            if not X_tr_clean.empty:
                model.fit(X_tr_clean, y_tr_clean)
                y_pred = model.predict(X_val)

                metrics['mse'].append(mean_squared_error(y_val, y_pred))
                metrics['r2'].append(r2_score(y_val, y_pred))
                metrics['mae'].append(mean_absolute_error(y_val, y_pred))

                if calculate_ecm:
                    p_val = price.loc[y_val.index]
                    p_val_clean = p_val.dropna()
                    y_val_clean = y_val.loc[p_val_clean.index]
                    y_pred_clean = pd.Series(y_pred, index=y_val.index).loc[p_val_clean.index] # Alinear predicciones

                    if not p_val_clean.empty:
                         metrics['ecm'].append(((y_val_clean - y_pred_clean).abs() * p_val_clean).mean())
                    else:
                         metrics['ecm'].append(np.nan)
                else:
                     metrics['ecm'].append(np.nan)
            else:
                 st.warning("Advertencia: No hay datos de entrenamiento limpios para este split de evaluación.")
                 metrics['mse'].append(np.nan)
                 metrics['r2'].append(np.nan)
                 metrics['mae'].append(np.nan)
                 metrics['ecm'].append(np.nan)


        results = {}
        for k,v in metrics.items():
            v_array = np.array(v)
            results[f'{k}_mean'] = float(np.nanmean(v_array)) if not np.all(np.isnan(v_array)) else np.nan
            results[f'{k}_std'] = float(np.nanstd(v_array)) if not np.all(np.isnan(v_array)) else np.nan
        return results


    def predict_future(self):
        """
        Realiza predicciones de consumo, generación y precio
        para el período futuro (`self.df_future`).
        """
        if self.df_future is None or self.df_future.empty:
            st.warning("No hay datos futuros disponibles para predecir.")
            return

        if 'hour' not in self.df_future.columns:
             self.df_future['hour'] = self.df_future['timestamp'].dt.hour
             self.df_future['dayweek'] = self.df_future['timestamp'].dt.weekday
             self.df_future['month'] = self.df_future['timestamp'].dt.month
             st.info("Features de tiempo creadas en datos futuros.")


        st.info("Realizando predicciones para el período futuro...")

        # Predecimos el consumo
        if self.model_consumo and all(f in self.df_future.columns for f in self.features_consumo):
            X_future_c = self.df_future[self.features_consumo]
            if not X_future_c.empty:
                self.df_future['consumo_pred'] = self.model_consumo.predict(X_future_c)
                self.df_future['consumo_pred'] = self.df_future['consumo_pred'].apply(lambda x: max(0, x))
                st.success("Predicción de consumo completada.")
            else:
                st.warning("No hay datos futuros limpios para predecir consumo.")
                self.df_future['consumo_pred'] = np.nan
        else:
            st.warning("Modelo de consumo no disponible o faltan columnas para predecir consumo en datos futuros.")
            self.df_future['consumo_pred'] = np.nan


        # Predecimos la generación solar
        if self.model_generacion and all(f in self.df_future.columns for f in self.features_generacion):
            X_future_g = self.df_future[self.features_generacion]
            if not X_future_g.empty:
                self.df_future['generacion_solar_pred'] = self.model_generacion.predict(X_future_g)
                self.df_future['generacion_solar_pred'] = self.df_future['generacion_solar_pred'].apply(lambda x: max(0, x))
                st.success("Predicción de generación solar completada.")
            else:
                st.warning("No hay datos futuros limpios para predecir generación.")
                self.df_future['generacion_solar_pred'] = np.nan
        else:
            st.warning("Modelo de generación no disponible o faltan columnas para predecir generación en datos futuros.")
            self.df_future['generacion_solar_pred'] = np.nan

        # Predecimos el precio de venta
        if self.model_precio and all(f in self.df_future.columns for f in self.features_precio):
            X_future_p = self.df_future[self.features_precio]
            if not X_future_p.empty:
                self.df_future['precio_venta_pred'] = self.model_precio.predict(X_future_p)
                self.df_future['precio_venta_pred'] = self.df_future['precio_venta_pred'].apply(lambda x: max(0, x))
                st.success("Predicción de precio completada.")
            else:
                st.warning("No hay datos futuros limpios para predecir precio.")
                self.df_future['precio_venta_pred'] = np.nan
        else:
            st.warning("Modelo de precio no disponible o faltan columnas para predecir precio en datos futuros.")
            self.df_future['precio_venta_pred'] = np.nan


        st.info("Predicciones futuras completadas.")


    def optimize_energy(self, bateria_max_kwh: float = 10.0, precio_compra_kw: float = 0.4, eficiencia_bateria: float = 0.95):
        """
        Define y resuelve el problema de optimización lineal para la gestión
        de energía con batería.

        Args:
            bateria_max_kwh: Capacidad máxima de la batería (kWh).
            precio_compra_kw: Precio de compra de electricidad (€/kWh).
            eficiencia_bateria: Eficiencia de carga/descarga de la batería.
        """
        if self.df_future is None or self.df_future.empty or 'consumo_pred' not in self.df_future.columns or 'generacion_solar_pred' not in self.df_future.columns or 'precio_venta_pred' not in self.df_future.columns:
            st.warning("No hay datos futuros con predicciones completas para la optimización.")
            return

        df_opt = self.df_future.dropna(subset=['consumo_pred', 'generacion_solar_pred', 'precio_venta_pred']).copy()

        if df_opt.empty:
            st.warning("No hay datos futuros limpios (sin NaN en predicciones) para la optimización.")
            return

        st.info("Iniciando el problema de optimización lineal...")

        T = len(df_opt)
        df_opt = df_opt.reset_index(drop=True)

        problem = LpProblem("Optimizacion_Energia_Hogar", LpMaximize)

        s = [LpVariable(f"s_{t}", lowBound=0, upBound=bateria_max_kwh) for t in range(T)]
        v = [LpVariable(f"v_{t}", lowBound=0) for t in range(T)]
        c = [LpVariable(f"c_{t}", lowBound=0) for t in range(T)]
        b_charge = [LpVariable(f"b_charge_{t}", lowBound=0) for t in range(T)]
        b_discharge = [LpVariable(f"b_discharge_{t}", lowBound=0) for t in range(T)]

        problem += lpSum([
            v[t] * df_opt['precio_venta_pred'].iloc[t]
            - c[t] * precio_compra_kw
            for t in range(T)
        ]), "Beneficio_Neto"

        for t in range(T):
            consumo_t = df_opt['consumo_pred'].iloc[t]
            generacion_t = df_opt['generacion_solar_pred'].iloc[t]

            problem += (
                generacion_t + b_discharge[t] + c[t] == consumo_t + b_charge[t] + v[t]
            ), f"Balance_Energia_{t}"

            if t == 0:
                problem += (
                    s[t] == 0 + b_charge[t] * eficiencia_bateria - b_discharge[t] * (1 / eficiencia_bateria)
                ), f"Bateria_Estado_{t}"
            else:
                problem += (
                    s[t] == s[t-1] + b_charge[t] * eficiencia_bateria - b_discharge[t] * (1 / eficiencia_bateria)
                ), f"Bateria_Estado_{t}"

            problem += (s[t] <= bateria_max_kwh), f"Bateria_Capacidad_{t}"
            problem += (b_charge[t] <= generacion_t), f"Bateria_LimiteCarga_Gen_{t}"
            problem += (b_charge[t] * eficiencia_bateria <= bateria_max_kwh - (s[t-1] if t > 0 else 0)), f"Bateria_LimiteCarga_Cap_{t}"
            problem += (b_discharge[t] <= (s[t-1] if t > 0 else 0) * eficiencia_bateria), f"Bateria_LimiteDescarga_{t}"


        st.info("Resolviendo el problema de optimización...")
        try:
            problem.solve()
            st.info(f"Status (Código): {LpStatus[problem.status]}")
            st.info(f"Beneficio total (Función Objetivo): {problem.objective.value():.2f} €")

            if LpStatus[problem.status] == 'Optimal':
                df_opt['bateria'] = [s[t].varValue for t in range(T)]
                df_opt['venta'] = [v[t].varValue for t in range(T)]
                df_opt['compra'] = [c[t].varValue for t in range(T)]
                df_opt['carga_bateria'] = [b_charge[t].varValue for t in range(T)]
                df_opt['descarga_bateria'] = [b_discharge[t].varValue for t in range(T)]
                st.success("Resultados de la optimización obtenidos.")

                self.df_future = pd.merge(self.df_future, df_opt[['timestamp', 'bateria', 'venta', 'compra', 'carga_bateria', 'descarga_bateria']],
                                          on='timestamp', how='left')
                cols_to_fill_zero = ['bateria', 'venta', 'compra', 'carga_bateria', 'descarga_bateria']
                for col in cols_to_fill_zero:
                     if col in self.df_future.columns:
                          self.df_future[col] = self.df_future[col].fillna(0)

                st.info("Resultados de optimización integrados en df_future.")

            else:
                st.warning("El problema de optimización no se resolvió de forma óptima.")
                cols_to_add_nan = ['bateria', 'venta', 'compra', 'carga_bateria', 'descarga_bateria']
                for col in cols_to_add_nan:
                     if col not in self.df_future.columns:
                          self.df_future[col] = np.nan

        except Exception as e:
            st.error(f"Error al resolver el problema de optimización: {e}")
            cols_to_add_nan = ['bateria', 'venta', 'compra', 'carga_bateria', 'descarga_bateria']
            for col in cols_to_add_nan:
                 if col not in self.df_future.columns:
                      self.df_future[col] = np.nan


    def analyze_results(self, precio_compra_kw: float = 0.4):
        """
        Calcula y muestra el beneficio/ahorro comparando diferentes escenarios.

        Args:
            precio_compra_kw: Precio de compra de electricidad (€/kWh) usado en el cálculo.
        """
        if self.df_future is None or self.df_future.empty or 'consumo_pred' not in self.df_future.columns or 'venta' not in self.df_future.columns:
            st.warning("No hay resultados de optimización disponibles para analizar.")
            return

        st.subheader("Análisis de Resultados")

        required_cols = ['venta', 'precio_venta_pred', 'compra']
        if not all(col in self.df_future.columns for col in required_cols):
             st.warning("Faltan columnas necesarias en df_future para calcular el beneficio/pérdida optimizado.")
             beneficio_total_optimizado = np.nan
        else:
             self.df_future['beneficio_hora'] = (
                 self.df_future['venta'].fillna(0) * self.df_future['precio_venta_pred'].fillna(0)
                 - self.df_future['compra'].fillna(0) * precio_compra_kw
             )
             beneficio_total_optimizado = self.df_future['beneficio_hora'].sum()


        if 'consumo_pred' in self.df_future.columns:
             gasto_total_sin_sistema = self.df_future['consumo_pred'].fillna(0).sum() * precio_compra_kw
        else:
             st.warning("Falta la columna 'consumo_pred' para calcular el gasto sin sistema.")
             gasto_total_sin_sistema = np.nan


        if 'consumo_pred' in self.df_future.columns and 'generacion_solar_pred' in self.df_future.columns and 'precio_venta_pred' in self.df_future.columns:
            consumo_pred_clean = self.df_future['consumo_pred'].fillna(0)
            generacion_solar_pred_clean = self.df_future['generacion_solar_pred'].fillna(0)
            precio_venta_pred_clean = self.df_future['precio_venta_pred'].fillna(0)

            energia_comprada_sin_baterias = np.maximum(consumo_pred_clean - generacion_solar_pred_clean, 0)
            energia_vendida_sin_baterias = np.maximum(generacion_solar_pred_clean - consumo_pred_clean, 0)

            costo_compra_sin_baterias = energia_comprada_sin_baterias * precio_compra_kw
            ingreso_venta_sin_baterias = energia_vendida_sin_baterias * precio_venta_pred_clean
            neto_sin_baterias = costo_compra_sin_baterias - ingreso_venta_sin_baterias
            gasto_total_sin_baterias = neto_sin_baterias.sum()
        else:
            st.warning("Faltan columnas necesarias para calcular el gasto solo con placas solares.")
            gasto_total_sin_baterias = np.nan


        ahorro_placas = gasto_total_sin_sistema - gasto_total_sin_baterias if not (np.isnan(gasto_total_sin_sistema) or np.isnan(gasto_total_sin_baterias)) else np.nan
        ahorro_baterias = gasto_total_sin_baterias - (-beneficio_total_optimizado) if not (np.isnan(gasto_total_sin_baterias) or np.isnan(beneficio_total_optimizado)) else np.nan
        ahorro_total = ahorro_placas + ahorro_baterias if not (np.isnan(ahorro_placas) or np.isnan(ahorro_baterias)) else np.nan


        st.write(f"**Beneficio/Pérdida total con batería (optimizado):** {beneficio_total_optimizado:.2f} €" if not np.isnan(beneficio_total_optimizado) else "**Beneficio/Pérdida total con batería:** No calculado")
        st.write(f"**Gasto total si no tuviera ningún sistema (100% compra):** {gasto_total_sin_sistema:.2f} €" if not np.isnan(gasto_total_sin_sistema) else "**Gasto total sin sistema:** No calculado")
        st.write(f"**Gasto total si tuviera solo placas solares (sin batería):** {gasto_total_sin_baterias:.2f} €" if not np.isnan(gasto_total_sin_baterias) else "**Gasto total solo con placas:** No calculado")
        st.markdown("---")
        st.write(f"**Ahorro por tener placas solares (sin batería):** {ahorro_placas:.2f} €" if not np.isnan(ahorro_placas) else "**Ahorro por placas:** No calculado")
        st.write(f"**Ahorro adicional por tener batería:** {ahorro_baterias:.2f} €" if not np.isnan(ahorro_baterias) else "**Ahorro adicional por batería:** No calculado")
        st.write(f"**Ahorro total:** {ahorro_total:.2f} €" if not np.isnan(ahorro_total) else "**Ahorro total:** No calculado")

        if self.df_future is not None and not self.df_future.empty:
            st.subheader("Resultados de la Optimización por Hora")
            st.dataframe(self.df_future[['timestamp', 'consumo_pred', 'generacion_solar_pred', 'precio_venta_pred', 'bateria', 'venta', 'compra', 'carga_bateria', 'descarga_bateria', 'beneficio_hora']].head())

            st.subheader("Visualización del Flujo de Energía")
            chart_data = self.df_future[['timestamp', 'consumo_pred', 'generacion_solar_pred', 'bateria', 'venta', 'compra']]
            chart_data = chart_data.set_index('timestamp')
            st.line_chart(chart_data)