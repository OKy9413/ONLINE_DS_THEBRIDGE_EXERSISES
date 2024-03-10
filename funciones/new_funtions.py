import pandas as pd
import numpy as np
import re
from bs4 import BeautifulSoup
import requests
import time
import random
import optuna
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score
from concurrent.futures import ThreadPoolExecutor, TimeoutError

def camel_to_snake(name):
    """
    Convierte cadenas a snake_case, con manejo especial para secuencias de letras mayúsculas e iniciales.
    Secuencias de letras mayúsculas se mantienen como un bloque si son seguidas por un guión bajo, un número, el final de la cadena, o letras minúsculas.
    Palabras en camelCase se dividen en palabras separadas.
    
    Parámetros:
    - name (str): La cadena a convertir.
    
    Devoluciones:
    - str: La cadena convertida en snake_case.
    """
    # Maneja secuencias de letras mayúsculas que son parte de iniciales o palabras separadas
    # Coincide con secuencias de letras mayúsculas seguidas por '_', un número, el final de la cadena, o letras minúsculas
    name = re.sub(r'([A-Z]+)([A-Z][a-z]|(?=_|\d|$))', r'\1\2', name)
    
    # Inserta un guión bajo antes de letras mayúsculas que son parte de palabras en camelCase, excluyendo el inicio de la cadena
    name = re.sub(r'(?<!^)(?=[A-Z][a-z])', '_', name).lower()
    
    return name



def rename_columns(df):
    """
    Renombra todas las columnas de un DataFrame a snake_case, con manejo avanzado para secuencias de letras mayúsculas.
    
    Parámetros:
    - df (pd.DataFrame): El DataFrame original.
    
    Devoluciones:
    - pd.DataFrame: Un nuevo DataFrame con las columnas renombradas a snake_case avanzado.
    """
    df_copy = df.copy()
    df_copy.columns = [camel_to_snake(col) for col in df.columns]
    return df_copy 



def replace_rare_otro(df, column_names, threshold=1):
    """
    Reemplaza los valores en una o varias columnas de un DataFrame que aparecen menos o igual que 'threshold' veces por 'otro'.
    
    Parámetros:
    - df (pd.DataFrame): El DataFrame original.
    - column_names (str or list): Un nombre de columna o una lista de nombres de columnas en las que buscar los valores raros.
    - threshold (int): El umbral de frecuencia por debajo o igual al cual los valores se consideran raros.
    
    Devoluciones:
    - pd.DataFrame: El DataFrame con los valores raros reemplazados.
    """
    # Si column_names es un string, lo convierte en una lista con un solo elemento.
    if isinstance(column_names, str):
        column_names = [column_names]
    
    for column_name in column_names:
        value_counts = df[column_name].value_counts()
        rare_values = value_counts[value_counts <= threshold].index.tolist()
        df[column_name] = df[column_name].apply(lambda x: 'otro' if x in rare_values else x)
    return df



def extract_resolution(df, column_name, divide=False):
    """
    Extrae la resolución con formato '0000x0000' de una columna especificada y la coloca en una nueva columna 'resolution',
    o en dos nuevas columnas 'resolution_x' y 'resolution_y' si divide es True. En los casos donde no se encuentra
    una resolución, asigna el valor 'otro'.
    
    Parámetros:
    - df (pd.DataFrame): El DataFrame original.
    - column_name (str): El nombre de la columna de la que extraer la resolución.
    - divide (bool): Si es True, divide la resolución en dos columnas ('resolution_x' y 'resolution_y').
    
    Devoluciones:
    - pd.DataFrame: El DataFrame modificado con la nueva columna o columnas.
    """
    if divide:
        # Extrae y divide la resolución en dos columnas
        df['resolution_x'], df['resolution_y'] = zip(*df[column_name].apply(lambda x: re.search(r'(\d{3,4})x(\d{3,4})', x).groups() if re.search(r'(\d{3,4})x(\d{3,4})', x) else ('otro', 'otro')))
    else:
        # Extrae la resolución en una columna
        df['resolution'] = df[column_name].apply(lambda x: re.search(r'\b\d{3,4}x\d{3,4}\b', x).group(0) if re.search(r'\b\d{3,4}x\d{3,4}\b', x) else 'otro')
    
    return df



def new_col_bool(df, column_name, terms):
    """
    Crea nuevas columnas en el DataFrame para marcar la presencia de términos específicos en una columna dada.
    
    Parámetros:
    - df (pd.DataFrame): El DataFrame original.
    - column_name (str): El nombre de la columna donde buscar los términos.
    - terms (list or str): Un término o una lista de términos a buscar en la columna.
    
    Devoluciones:
    - pd.DataFrame: El DataFrame con las nuevas columnas añadidas.
    """
    # Asegura que 'terms' sea una lista para facilitar la iteración
    if isinstance(terms, str):
        terms = [terms]
    
    for term in terms:
        # Crea un nombre de columna válido reemplazando espacios con '_'
        new_column_name = term.replace(" ", "_")
        # Usa str.contains para buscar el término en la columna especificada, marcando con 1 las filas que contienen el término
        df[new_column_name] = df[column_name].str.contains(term, case=False, na=False).astype(int)

    return df



def extract_cpu_ghz(df, column_name):
    """
    Parsea los valores de GHz de la columna especificada, aceptando un rango más amplio de formatos numéricos,
    y los coloca en una nueva columna 'cpu_GHz' como valores numéricos.
    
    Parámetros:
    - df (pd.DataFrame): El DataFrame original.
    - column_name (str): El nombre de la columna de la que extraer los GHz.
    """
    # Define una función para extraer los valores numéricos de los GHz, aceptando un rango más amplio de formatos numéricos
    def parse_ghz(value):
        match = re.search(r'(\d+(\.\d+)?)GHz', value)
        return float(match.group(1)) if match else None

    # Aplica la función a la columna especificada y crea la nueva columna 'cpu_GHz'
    df['cpu_GHz'] = df[column_name].apply(parse_ghz)



def create_binary_cpu_columns(df, column_name):
    """
    Crea dos columnas binarias en el DataFrame para indicar si el procesador es AMD o Intel.
    
    Parámetros:
    - df (pd.DataFrame): El DataFrame original.
    - column_name (str): El nombre de la columna de la que determinar si el procesador es AMD o Intel.
    """
    # Crea una columna binaria para AMD
    df['cpu_AMD'] = df[column_name].str.contains('AMD', case=False, na=False).astype(int)
    
    # Crea una columna binaria para Intel
    df['cpu_Intel'] = df[column_name].str.contains('Intel', case=False, na=False).astype(int)



def extract_ram_gb(df, column_name):
    """
    Extrae la cantidad de memoria RAM de la columna especificada y la coloca en una nueva columna 'ram_gb' como valores enteros.
    
    Parámetros:
    - df (pd.DataFrame): El DataFrame original.
    - column_name (str): El nombre de la columna de la que extraer la cantidad de RAM.
    """
    # Define una función para extraer los valores enteros de la cantidad de RAM
    def parse_ram(value):
        match = re.search(r'(\d+)', value)
        return int(match.group(1)) if match else None

    # Aplica la función a la columna especificada y crea la nueva columna 'ram_gb'
    df['ram_gb'] = df[column_name].apply(parse_ram)



def classify_memory_type(df, column_name):
    """
    Crea dos columnas binarias en el DataFrame para indicar si la memoria es HDD o SSD.
    
    Parámetros:
    - df (pd.DataFrame): El DataFrame original.
    - column_name (str): El nombre de la columna de la que determinar el tipo de memoria.
    """
    # Crea una columna binaria para HDD
    df['hdd_memory'] = df[column_name].str.contains('HDD', case=False, na=False).astype(int)
    
    # Crea una columna binaria para SSD
    df['ssd_memory'] = df[column_name].str.contains('SSD', case=False, na=False).astype(int)



def convert_memory_to_gb(df, column_name):
    """
    Detecta las unidades de memoria (GB o TB) en la columna especificada, convierte los valores numéricos a GB,
    y coloca los valores convertidos en una nueva columna llamada 'memory_GB'.
    
    Parámetros:
    - df (pd.DataFrame): El DataFrame original.
    - column_name (str): El nombre de la columna de la que convertir los valores de memoria a GB.
    """
    def parse_memory(value):
        # Busca valores numéricos seguidos de GB o TB (ignorando mayúsculas/minúsculas)
        match_gb = re.search(r'(\d+)(GB)', value, re.IGNORECASE)
        match_tb = re.search(r'(\d+)(TB)', value, re.IGNORECASE)
        
        # Convierte a GB si encuentra GB directamente
        if match_gb:
            return int(match_gb.group(1))
        # Convierte a GB multiplicando por 1000 si encuentra TB
        elif match_tb:
            return int(match_tb.group(1)) * 1000
        else:
            return None

    # Aplica la función de conversión a la columna especificada y crea la nueva columna 'memory_GB'
    df['memory_gb'] = df[column_name].apply(parse_memory)



def modify_cpu_model_names(df, column_name):
    """
    Modifica directamente el DataFrame dado, quitando las dos últimas palabras y la primera palabra
    de cada valor en la columna especificada, y almacena el resultado en una nueva columna 'cpu_model'.
    """
    df['cpu_model'] = df[column_name].apply(lambda x: " ".join(x.split()[1:-2]))



def create_binary_gpu_columns(df, column_name):
    """
    Crea tres columnas binarias en el DataFrame para indicar si la GPU es de Nvidia, AMD o Intel.
    
    Parámetros:
    - df (pd.DataFrame): El DataFrame original.
    - column_name (str): El nombre de la columna de la que determinar si la GPU es de Nvidia, AMD o Intel.
    """
    # Crea una columna binaria para Nvidia
    df['gpu_Nvidia'] = df[column_name].str.contains('Nvidia', case=False, na=False).astype(int)
    
    # Crea una columna binaria para AMD
    df['gpu_AMD'] = df[column_name].str.contains('AMD', case=False, na=False).astype(int)
    
    # Crea una columna binaria para Intel
    df['gpu_Intel'] = df[column_name].str.contains('Intel', case=False, na=False).astype(int)




def remove_first_word_and_create_new_column(df, column_name):
    """
    Quita la primera palabra de cada valor en la columna especificada y almacena el resultado
    en una nueva columna llamada 'gpu_spec' dentro del mismo DataFrame, modificándolo directamente.
    
    Parámetros:
    - df (pd.DataFrame): El DataFrame original a modificar.
    - column_name (str): El nombre de la columna de la que quitar la primera palabra.
    """
    df['gpu_spec'] = df[column_name].apply(lambda x: " ".join(x.split()[1:]))



def remove_kg(df, column_name):
    """
    Modifica directamente la columna especificada del DataFrame, quitando 'kg' de cada valor y convirtiéndolo a float.
    
    Parámetros:
    - df (pd.DataFrame): El DataFrame original a modificar.
    - column_name (str): El nombre de la columna donde cada valor termina con 'kg'.
    """
    df[column_name] = df[column_name].str.replace('kg', '', case=False).astype(float)



def apply_one_hot_encoding(df, columns_list):
    """
    Aplica One-Hot Encoding a las columnas especificadas en el DataFrame, convierte los valores a enteros,
    y elimina las columnas originales, asegurando que los resultados sean 0 y 1.
    
    Parámetros:
    - df (pd.DataFrame): El DataFrame original a modificar.
    - columns_list (list): Lista de nombres de columnas a las que aplicar One-Hot Encoding.
    """
    for column in columns_list:
        # Aplica One-Hot Encoding y asegura que los resultados sean enteros
        dummies = pd.get_dummies(df[column], prefix=column, drop_first=False).astype(int)
        df = pd.concat([df, dummies], axis=1)
        df.drop(column, axis=1, inplace=True)
    
    return df



def optimize_hyperparams_rf_optuna(X, y):
    """
    Optimiza hiperparámetros para RandomForestRegressor usando Optuna.
    
    Argumentos:
    X -- Conjunto de características
    y -- Vector objetivo
    
    Devuelve:
    best_params -- Los mejores hiperparámetros encontrados
    """
    def objective_rf(trial):
        # Ampliar el espacio de búsqueda de hiperparámetros
        n_estimators = trial.suggest_int('n_estimators', 100, 1000)  # Ampliado a 100-1000
        max_depth = trial.suggest_int('max_depth', 10, 100)  # Ampliado a 10-100
        min_samples_split = trial.suggest_int('min_samples_split', 2, 50)  # Ampliado a 2-50
        min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 20)  # Ampliado a 1-20
        
        # Instanciar y entrenar el modelo con los hiperparámetros sugeridos
        model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            random_state=42
        )
        
        # Usar cross-validation para evaluar el modelo
        score = cross_val_score(model, X, y, n_jobs=-1, cv=5, scoring='neg_mean_squared_error').mean()
        
        return score

    study = optuna.create_study(direction='maximize')
    study.optimize(objective_rf, n_trials=50)  # Aumentado a 50 trials
    
    print("Optimización con Optuna para RandomForestRegressor completada.")
    print(f"Mejores hiperparámetros: {study.best_params}")
    print(f"Mejor score: {study.best_value}\n")
    
    return study.best_params
    


def optimize_hyperparams_dt_optuna(X, y):
    """
    Optimiza hiperparámetros para DecisionTreeRegressor usando Optuna.
    
    Argumentos:
    X -- Conjunto de características
    y -- Vector objetivo
    
    Devuelve:
    best_params -- Los mejores hiperparámetros encontrados
    """
    def objective_dt(trial):
        # Definir el espacio de búsqueda de hiperparámetros
        max_depth = trial.suggest_int('max_depth', 5, 50)
        min_samples_split = trial.suggest_int('min_samples_split', 2, 20)
        min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 10)
        
        model = DecisionTreeRegressor(
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            random_state=42
        )
        
        score = cross_val_score(model, X, y, n_jobs=-1, cv=5, scoring='neg_mean_squared_error').mean()
        
        return score

    study = optuna.create_study(direction='maximize')
    study.optimize(objective_dt, n_trials=30)
    
    print("Optimización con Optuna para DecisionTreeRegressor completada.")
    print(f"Mejores hiperparámetros: {study.best_params}")
    print(f"Mejor score: {study.best_value}\n")
    
    return study.best_params



def optimize_hyperparams_svr_optuna(X, y):
    """
    Optimiza hiperparámetros para SVR (Support Vector Regression) usando Optuna, con un límite de tiempo de 1 minuto por trial.
    
    Argumentos:
    X -- Conjunto de características
    y -- Vector objetivo
    
    Devuelve:
    best_params -- Los mejores hiperparámetros encontrados
    """
    def objective_svr(trial):
        C = trial.suggest_loguniform('C', 1e-10, 1e10)
        kernel = trial.suggest_categorical('kernel', ['linear', 'poly', 'rbf', 'sigmoid'])
        gamma = trial.suggest_categorical('gamma', ['scale', 'auto'])
        degree = trial.suggest_int('degree', 1, 5) if kernel == 'poly' else 3

        def evaluate_model():
            model = SVR(C=C, kernel=kernel, gamma=gamma, degree=degree)
            scores = cross_val_score(model, X, y, n_jobs=-1, cv=5, scoring='neg_mean_squared_error')
            rmse_scores = np.sqrt(-scores)
            return rmse_scores.mean()
        
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(evaluate_model)
            try:
                return future.result(timeout=60)  # 60 segundos de tiempo límite
            except TimeoutError:
                return np.inf  # Retornar un valor que indique un mal rendimiento

    study = optuna.create_study(direction='minimize')
    study.optimize(objective_svr, n_trials=50,n_jobs = -1)
    
    print("Optimización con Optuna para SVR completada.")
    print(f"Mejores hiperparámetros: {study.best_params}")
    print(f"Mejor RMSE: {study.best_value}\n")
    
    return study.best_params



def buscar_valor_mas_parecido_por_producto_y_tipo(dataset, producto_parcial, tipo):
    """
    Busca en el dataset el producto que más se asemeje a una subcadena dada y tipo específico,
    y devuelve el Process Size (nm) correspondiente.

    Parámetros:
    - dataset: DataFrame de Pandas que contiene los datos.
    - producto_parcial: Subcadena del nombre del producto a buscar.
    - tipo: El tipo de producto ('CPU' o 'GPU').

    Devuelve:
    - El valor de 'Process Size (nm)' del producto que más se asemeje a la subcadena dada y tipo especificado.
    - None si no se encuentra ningún producto que coincida con los criterios.
    """
    # Filtrar el DataFrame por el tipo especificado
    resultado = dataset[dataset['Type'] == tipo]
    
    # Calcular un puntaje de coincidencia basado en la cantidad de palabras en común
    # (esto es una simplificación y podría ser reemplazado por un cálculo de similitud más sofisticado)
    resultado['Coincidencia'] = resultado['Product'].apply(lambda x: sum(word in x.lower() for word in producto_parcial.lower().split()))

    # Encontrar el índice del producto con mayor puntaje de coincidencia
    idx_max = resultado['Coincidencia'].idxmax()

    # Verificar si hay al menos una coincidencia
    if resultado.loc[idx_max, 'Coincidencia'] > 0:
        # Devolver el valor de 'Process Size (nm)' para el producto con mayor coincidencia
        return resultado.loc[idx_max, 'Process Size (nm)']
    else:
        # Devolver None si no se encontró ninguna coincidencia
        return None
    


def agregar_process_size_directamente(df, columnas_dict, dataset):
    """
    Modifica directamente el DataFrame dado, agregando los valores de 'Process Size (nm)' para los productos
    CPU y GPU especificados en las columnas dadas por el diccionario. Los resultados se añaden en las nuevas
    columnas 'cpu_ps' y 'gpu_ps'.

    Parámetros:
    - df: DataFrame de Pandas que será modificado directamente.
    - columnas_dict: Diccionario con las claves 'CPU' o 'GPU' y los valores son los nombres de las columnas
      en df donde se encuentran esos valores.
    - dataset: DataFrame que contiene la información de 'Process Size (nm)' para cada producto.
    """
    # Inicializar las nuevas columnas con NaNs en el DataFrame original
    df['cpu_ps'] = pd.NA
    df['gpu_ps'] = pd.NA
    
    for tipo, columna in columnas_dict.items():
        # Asegurar que tipo es 'CPU' o 'GPU'
        if tipo in ['CPU', 'GPU']:
            # Recorrer cada valor en la columna especificada
            for index, valor in df[columna].items():
                # Crear un DataFrame temporal para evitar SettingWithCopyWarning
                temp_df = dataset[dataset['Type'] == tipo].copy()
                temp_df['Coincidencia'] = temp_df['Product'].apply(lambda x: sum(word in x.lower() for word in valor.lower().split()))

                # Encontrar el índice del producto con mayor puntaje de coincidencia
                idx_max = temp_df['Coincidencia'].idxmax()

                # Verificar si hay al menos una coincidencia
                if temp_df.loc[idx_max, 'Coincidencia'] > 0:
                    # Asignar el valor encontrado a la columna correspondiente en df
                    if tipo == 'CPU':
                        df.at[index, 'cpu_ps'] = temp_df.loc[idx_max, 'Process Size (nm)']
                    elif tipo == 'GPU':
                        df.at[index, 'gpu_ps'] = temp_df.loc[idx_max, 'Process Size (nm)']