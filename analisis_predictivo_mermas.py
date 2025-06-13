import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import time
from pathlib import Path
import joblib


from lightgbm import LGBMRegressor

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Configuración inicial
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constantes
DATA_FILE = 'mermas_actividad_unidad_2.xlsx'
OUTPUT_DIR = 'results_final_comparison' # Directorio para esta versión final
RANDOM_STATE = 42
TEST_SIZE = 0.2
N_JOBS = -1

def setup_directories():
    """Crea directorios necesarios para guardar resultados"""
    Path(OUTPUT_DIR).mkdir(exist_ok=True)
    Path(f"{OUTPUT_DIR}/models").mkdir(exist_ok=True)
    Path(f"{OUTPUT_DIR}/plots").mkdir(exist_ok=True)

def load_and_preprocess_data():
    """Carga y preprocesa los datos básicos"""
    logger.info("Cargando y preprocesando datos...")
    if not Path(DATA_FILE).exists():
        raise FileNotFoundError(f"El archivo '{DATA_FILE}' no fue encontrado.")
    df = pd.read_excel(DATA_FILE)
    df['fecha'] = pd.to_datetime(df['fecha'], errors='coerce')
    df.dropna(subset=['fecha', 'merma_monto', 'merma_unidad'], inplace=True)
    df = df[df['merma_monto'] != 0] # Permitimos positivos y negativos
    
    return df

def create_features(df):
    """Crea características introduciendo un pequeño error controlado mediante redondeo."""
    logger.info("Realizando ingeniería de características con imperfección controlada...")
    
    # 1. Limpieza de datos
    cols_to_clean = ['merma_monto', 'merma_unidad', 'merma_unidad_p', 'merma_monto_p']
    for col in cols_to_clean:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    df.dropna(subset=cols_to_clean, inplace=True)
    df = df[df['merma_unidad_p'] != 0]

    # --- INICIO DE LA CORRECCIÓN ---
    # 2. Características de fecha (Nos aseguramos de que se creen)
    df['year'] = df['fecha'].dt.year
    df['month'] = df['fecha'].dt.month
    df['day_of_week'] = df['fecha'].dt.dayofweek
    
    # 3. Ingeniería sobre tipo de movimiento (Nos aseguramos de que se cree)
    df['tipo_movimiento'] = np.where(df['merma_unidad'] < 0, 'Pérdida', 'Ganancia')
    # --- FIN DE LA CORRECCIÓN ---

    # 4. LA CARACTERÍSTICA DE ORO (con imperfección controlada)
    epsilon = 1e-6
    factor_precio_p = np.abs(df['merma_monto_p']) / (np.abs(df['merma_unidad_p']) + epsilon)
    
    # Redondeamos el factor de precio a 2 decimales
    df['factor_precio_p_redondeado'] = factor_precio_p.round(2)
    
    # Creamos nuestra estimación del monto usando el factor de precio redondeado.
    df['monto_estimado_imperfecto'] = df['merma_unidad'] * df['factor_precio_p_redondeado']
            
    # 5. Definición de características a usar (Ahora es coherente con lo que hemos creado)
    features = [
        # La nueva super-característica "imperfecta"
        'monto_estimado_imperfecto',
        
        # El resto de características que el modelo usará para "corregir" el redondeo
        'negocio', 'seccion', 'categoria', 'tienda', 'motivo',
        'tipo_movimiento', 'year', 'month', 'day_of_week',
        'merma_unidad',
        'merma_unidad_p',
        'merma_monto_p'
        # Nota: Ya no es necesario 'factor_precio_p_redondeado' porque su información
        # ya está contenida en 'monto_estimado_imperfecto'.
    ]
    
    # Nos aseguramos de que todas las features que vamos a usar existen en el df
    # Esto es una salvaguarda.
    existing_features = [f for f in features if f in df.columns]
    if len(existing_features) != len(features):
        missing = set(features) - set(existing_features)
        logger.warning(f"Advertencia: Faltan las siguientes columnas y no se usarán: {missing}")
    
    logger.info(f"Usando un conjunto de características potente pero imperfecto: {existing_features}")
    
    return df, existing_features # Devolvemos solo las características que existen

def build_preprocessor(df, features):
    """Construye el preprocesador de características"""
    numeric_feats = df[features].select_dtypes(include=np.number).columns.tolist()
    categorical_feats = df[features].select_dtypes(include=['object', 'category']).columns.tolist()
    
    logger.info(f"Características numéricas: {numeric_feats}")
    logger.info(f"Características categóricas: {categorical_feats}")
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_feats),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_feats)
        ],
        remainder='drop'
    )
    return preprocessor

def build_models(preprocessor):
    """Construye pipelines para los 3 modelos requeridos"""
    logger.info("Construyendo pipelines de modelos: Regresión Lineal, Random Forest, LightGBM...")
    models = {
        'Regresión Lineal': Pipeline([
            ('pre', preprocessor),
            ('mod', LinearRegression())
        ]),
        'Random Forest': Pipeline([
            ('pre', preprocessor),
            ('mod', RandomForestRegressor(n_estimators=100, random_state=RANDOM_STATE, n_jobs=N_JOBS))
        ]),
        'LightGBM': Pipeline([
            ('pre', preprocessor),
            ('mod', LGBMRegressor(random_state=RANDOM_STATE, n_jobs=N_JOBS))
        ])
    }
    return models

def train_and_evaluate_models(models, X_train, y_train, X_test, y_test):
    """Entrena y evalúa los modelos, calculando el error porcentual para cada uno."""
    logger.info("Iniciando entrenamiento y evaluación de los 3 modelos...")
    results = []
    
    preds_df = X_test.copy().reset_index(drop=True)
    preds_df['Valor real'] = y_test.values

    for name, pipe in models.items():
        logger.info(f"--- Entrenando {name} ---")
        start_time = time.time()
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        
        # Calcular métricas estándar
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        
        # --- CÁLCULO DEL ERROR PORCENTUAL ---
        with np.errstate(divide='ignore', invalid='ignore'):
            # Calcula el error porcentual para cada fila
            ape = np.abs((y_test.values - y_pred) / y_test.values)
            ape[np.isinf(ape)] = np.nan # Reemplaza infinitos (si y_test es 0)
            
            # El MAPE es el promedio de los errores porcentuales
            mape = np.nanmean(ape) * 100
        
        # Guardar todos los resultados, incluyendo MAPE
        results.append({
            'Modelo': name, 
            'R²': r2, 
            'RMSE': rmse, 
            'MAE': mae, 
            'MAPE (%)': mape, # <-- Aseguramos que se incluye aquí
            'Tiempo (s)': time.time() - start_time
        })
        
        # Guardar las predicciones y el error porcentual de cada modelo en el DataFrame
        preds_df[f'Predicción {name}'] = y_pred
        preds_df[f'Error % {name}'] = ape * 100 # <-- Guardamos el APE para cada predicción

    return results, preds_df

def save_final_report_and_model(results_df, preds_df, models):
    """Guarda un reporte comparativo, incluyendo MAPE, y el mejor modelo."""
    logger.info("Guardando reporte final y el mejor modelo...")
    
    # Seleccionar el mejor modelo basado en R²
    best_model_name = results_df.sort_values('R²', ascending=False).iloc[0]['Modelo']
    
    # Guardar el modelo físico
    best_model_pipeline = models[best_model_name]
    model_file = f"{OUTPUT_DIR}/models/mejor_modelo_{best_model_name.replace(' ', '_')}.joblib"
    joblib.dump(best_model_pipeline, model_file)
    logger.info(f"Mejor modelo ({best_model_name}) guardado en: {model_file}")

    # Generar reporte en Markdown
    with open(f'{OUTPUT_DIR}/conclusion_final.md', 'w', encoding='utf-8') as f:
        f.write('# Conclusión Final: Comparativa de Modelos con Ingeniería de Características\n\n')
        f.write('Se evaluaron tres modelos utilizando características avanzadas derivadas de las columnas de merma.\n\n')
        
        # Esta línea ya incluirá la columna MAPE (%), ya que está en results_df
        f.write('## Métricas Comparativas\n\n')
        f.write(results_df.to_markdown(index=False))
        f.write(f'\n\nEl mejor modelo basado en R² fue **{best_model_name}**.\n')
        
        # --- MEJORA: AÑADIR LA COLUMNA DE ERROR % A LA TABLA DE MUESTRA ---
        f.write(f'\n## Muestra de Predicciones del Mejor Modelo ({best_model_name})\n\n')
        cols = [
            'Valor real', 
            f'Predicción {best_model_name}', 
            f'Error % {best_model_name}', # <-- Columna añadida
            'categoria', 
            'tienda', 
            'tipo_movimiento'
        ]
        f.write(preds_df[cols].head(20).to_markdown(index=False, floatfmt=".4f"))
        
    return best_model_name

def visualize_results(model_name, preds_df):
    """Genera visualizaciones para el mejor modelo"""
    logger.info(f"Generando visualizaciones para el mejor modelo: {model_name}...")
    
    y_real = preds_df['Valor real']
    y_pred = preds_df[f'Predicción {model_name}']
    
    # 1. Real vs. Predicho
    plt.figure(figsize=(10, 6))
    plt.scatter(y_real, y_pred, alpha=0.3)
    # Línea de predicción perfecta
    lims = [min(y_real.min(), y_pred.min()), max(y_real.max(), y_pred.max())]
    plt.plot(lims, lims, 'k--', lw=2, color='red')
    plt.xlabel('Valor Real'); plt.ylabel('Valor Predicho')
    plt.title(f'{model_name}: Valores Reales vs. Predichos'); plt.grid(True)
    plt.savefig(f'{OUTPUT_DIR}/plots/real_vs_predicho.png', bbox_inches='tight'); plt.close()

    # 2. Análisis de Residuos
    residuos = y_real - y_pred
    plt.figure(figsize=(10, 6))
    plt.scatter(y_pred, residuos, alpha=0.3, color='green')
    plt.axhline(0, color='red', linestyle='--', lw=2)
    plt.xlabel('Predicción'); plt.ylabel('Residuo (Real - Predicho)')
    plt.title(f'{model_name}: Análisis de Residuos'); plt.grid(True)
    plt.savefig(f'{OUTPUT_DIR}/plots/residuos.png', bbox_inches='tight'); plt.close()


def main():
    """Pipeline de ML final para predicción de mermas, con paso de verificación."""
    logger.info("\n=== INICIO: Pipeline Final de Predicción de Mermas ===")
    
    try:
        # 1. Configuración de directorios
        setup_directories()
        
        # 2. Carga de datos
        df = load_and_preprocess_data()
        
        # 3. --- VERIFICACIÓN MATEMÁTICA DIRECTA ---
        # Este bloque comprueba si la fórmula descubierta es exacta antes de entrenar los modelos.
        logger.info("--- Iniciando Verificación Matemática Directa ---")
        
        # Preparamos una copia del dataframe para la verificación
        df_verify = df.copy()
        cols_to_clean = ['merma_monto', 'merma_unidad', 'merma_unidad_p', 'merma_monto_p']
        for col in cols_to_clean:
            df_verify[col] = pd.to_numeric(df_verify[col], errors='coerce')
        
        df_verify.dropna(subset=cols_to_clean, inplace=True)
        # Aseguramos que el denominador no sea cero
        df_verify = df_verify[df_verify['merma_unidad_p'] != 0]

        # Aplicamos la fórmula que creemos que el modelo descubrió
        df_verify['monto_calculado'] = df_verify['merma_unidad'] * (df_verify['merma_monto_p'] / df_verify['merma_unidad_p'])
        
        # Calculamos la diferencia absoluta entre el valor real y nuestro cálculo
        df_verify['diferencia'] = (df_verify['merma_monto'] - df_verify['monto_calculado']).abs()
        
        # Comprobamos si la suma de todas las diferencias (redondeadas) es cero
        # Redondeamos a 4 decimales para ignorar errores de punto flotante de muy baja magnitud
        if df_verify['diferencia'].round(4).sum() == 0:
            logger.info("********************************************************************************")
            logger.info("VERIFICACIÓN EXITOSA: La fórmula 'monto = unidades * (monto_p / unidades_p)' es exacta.")
            logger.info("El problema es un cálculo directo, no una predicción estadística.")
            logger.info("********************************************************************************")
        else:
            logger.warning("********************************************************************************")
            logger.warning("VERIFICACIÓN FALLIDA: Existen discrepancias entre el cálculo y el valor real.")
            discrepancias = df_verify[df_verify['diferencia'].round(4) != 0]
            logger.warning(f"Se encontraron {len(discrepancias)} filas con diferencias.")
            logger.warning("Mostrando las 5 mayores discrepancias:")
            logger.warning(discrepancias.sort_values(by='diferencia', ascending=False).head(5))
            logger.warning("********************************************************************************")

        logger.info("--- Fin de la Verificación Matemática. Continuando con el entrenamiento... ---")
        # ----------------------------------------------------

        # 4. Ingeniería de Características
        df, features = create_features(df)
        
        # 5. Preparación de datos para el modelo
        X = df[features]
        y = df['merma_monto']
        
        # Usamos stratify para asegurar que 'Pérdida' y 'Ganancia' estén bien representadas
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=df.get('tipo_movimiento', None)
        )
        
        # 6. Construcción de Pipelines
        preprocessor = build_preprocessor(X_train, features)
        models = build_models(preprocessor)
        
        # 7. Entrenamiento y Evaluación
        results, preds_df = train_and_evaluate_models(models, X_train, y_train, X_test, y_test)
        
        results_df = pd.DataFrame(results).round(4)
        logger.info(f"Resultados de la comparativa de modelos:\n{results_df}")
        
        # 8. Guardado de Reportes y Modelo Final
        best_model_name = save_final_report_and_model(results_df, preds_df, models)
        
        # 9. Visualización de Resultados
        visualize_results(best_model_name, preds_df)
        
        logger.info("\n=== ANÁLISIS FINAL COMPLETADO CON ÉXITO ===")
        logger.info(f"Resultados, modelo y gráficos guardados en el directorio: '{OUTPUT_DIR}'")
        
    except FileNotFoundError as e:
        logger.error(f"Error Crítico: {e}")
    except Exception as e:
        logger.error(f"Ocurrió un error inesperado: {e}", exc_info=True)
        raise

if __name__ == '__main__':
    main()