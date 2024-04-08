from catboost import CatBoostClassifier, Pool
from collections import Counter
from scipy.stats import f_oneway, chi2_contingency
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif, SelectFromModel, RFE
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from mlxtend.feature_selection import SequentialFeatureSelector
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)


##########PRIMERA FUNCION: eval_model###########

def detect_problem_type(target):
    unique_values = np.unique(target)
    # Comprobación de si todos los valores son enteros o strings
    all_int_or_str = all(isinstance(x, (int, np.integer, str)) for x in unique_values)
    
    # Clasificará como 'clas' bajo las condiciones especificadas
    if len(unique_values) <= 5 or (len(unique_values) <= 15 and all_int_or_str):
        return "clas"
    else:
        return "reg"

def find_class_label(target_labels, class_label):
    """Busca en target_labels una coincidencia insensible a mayúsculas/minúsculas para class_label.
    Retorna la versión exacta encontrada en target_labels y un booleano indicando si hubo asimilación."""
    target_labels_lower = [label.lower() for label in target_labels]
    class_label_lower = class_label.lower()
    for original_label, lowered_label in zip(target_labels, target_labels_lower):
        if class_label_lower == lowered_label:
            return original_label, original_label.lower() != class_label_lower
    return None, False

def eval_reg(target, pred, metrics):
    """
    Evalúa y devuelve las métricas de rendimiento para problemas de regresión.

    Parámetros:
    - target : array-like, los verdaderos valores objetivo.
    - pred : array-like, las predicciones del modelo.
    - metrics : list, las métricas a calcular. Puede incluir 'RMSE', 'MAE', 'MAPE', 'GRAPH' o 'ALL'.

    Retorna:
    - Un diccionario con las métricas de regresión calculadas. Si se solicita una métrica no manejada,
      se incluirá en el diccionario con un valor de None y se imprimirá un mensaje al final.
    """
    results = {}
    unsupported_metrics = []

    if 'ALL' in metrics:
        metrics = ['RMSE', 'MAE', 'MAPE', 'GRAPH']
    elif isinstance(metrics, str):
        metrics = [metrics]

    for m in metrics:
        m_upper = m.upper()  # Normalizar métricas a mayúsculas

        if m_upper == "RMSE":
            rmse = np.sqrt(mean_squared_error(target, pred))
            print(f"RMSE: {rmse}")
            results["RMSE"] = rmse

        elif m_upper == "MAE":
            mae = mean_absolute_error(target, pred)
            print(f"MAE: {mae}")
            results["MAE"] = mae

        elif m_upper == "MAPE":
            try:
                mape = np.mean(np.abs((target - pred) / target)) * 100
                print(f"MAPE: {mape}%")
                results["MAPE"] = mape
            except ZeroDivisionError:
                print("MAPE cannot be calculated due to division by zero; returning MAE instead.")
                if "MAE" not in results:
                    mae = mean_absolute_error(target, pred)
                    print(f"MAE calculated instead: {mae}")
                    results["MAE"] = mae

        elif m_upper == "GRAPH":
            plt.figure(figsize=(6, 6))

            # Calcular los límites comunes para ambos ejes
            min_val = min(min(target), min(pred))
            max_val = max(max(target), max(pred))

            # Generar el scatter plot
            plt.scatter(target, pred)
            plt.xlabel('True Values')
            plt.ylabel('Predictions')
            plt.title('Comparison of True Values and Predictions')

            # Ajustar los límites de los ejes
            common_limits = [min_val, max_val]
            plt.xlim(common_limits)
            plt.ylim(common_limits)


            # Añadir una recta roja en diagonal que represente x = y
            plt.plot(common_limits, common_limits, color='red', linestyle='--')

        else:
            unsupported_metrics.append(m)

    if unsupported_metrics:
        print(f"Unsupported metrics requested: {', '.join(unsupported_metrics)}")
        for metric in unsupported_metrics:
            results[metric] = None

    return results

def eval_clas(target, pred, metrics, average='macro'):
    """
    Evalúa y devuelve las métricas de rendimiento para problemas de clasificación.

    Parámetros:
    - target : array-like, Verdaderos valores objetivo.
    - pred : array-like, Predicciones del modelo.
    - metrics : list, Lista de métricas a calcular. Puede incluir nombres de métricas generales y específicas de clase.
    - average: str, Método de promediado para métricas multiclase ('macro', 'micro', 'weighted', 'binary').

    Retorna:
    - Un diccionario con las métricas de clasificación calculadas.
    """
    results = {}
    unique_classes = np.unique(target)
    metrics_to_process = []

    if 'ALL' in metrics:
        metrics_to_process.extend(["ACCURACY", "PRECISION", "RECALL", "F1", "CLASS_REPORT", "MATRIX",
                                   "MATRIX_RECALL", "MATRIX_PRED"])
        metrics_to_process.extend(["PRECISION_" + str(u) for u in unique_classes] +
                                  ["RECALL_" + str(u) for u in unique_classes] +
                                  ["F1_" + str(u) for u in unique_classes])
    else:
        metrics_to_process.extend(metrics)

    for m in metrics_to_process:
        m_upper = m.upper()  # Asegura la insensibilidad a mayúsculas/minúsculas de las métricas

        if m_upper == "ACCURACY":
            accuracy = accuracy_score(target, pred)
            print(f"Accuracy: {accuracy}")
            results["ACCURACY"] = accuracy

        elif m_upper in ["PRECISION", "RECALL", "F1"]:
            if m_upper == "PRECISION":
                precision = precision_score(target, pred, average=average)
                print(f"Precision: {precision}")
                results["PRECISION"] = precision
            elif m_upper == "RECALL":
                recall = recall_score(target, pred, average=average)
                print(f"Recall: {recall}")
                results["RECALL"] = recall
            elif m_upper == "F1":
                f1 = f1_score(target, pred, average=average)
                print(f"F1 Score: {f1}")
                results["F1"] = f1

        elif m_upper == "CLASS_REPORT":
            report = classification_report(target, pred)
            print("Classification Report:\n", report)
            results["CLASS_REPORT"] = report

        elif m_upper == "MATRIX":
            cm = confusion_matrix(target, pred)
            ConfusionMatrixDisplay(cm).plot(values_format='d')
            plt.show()
            results["MATRIX"] = cm

        elif m_upper == "MATRIX_RECALL":
            cm_recall = confusion_matrix(target, pred, normalize='true')
            ConfusionMatrixDisplay(cm_recall).plot(values_format='.2f', cmap='viridis')
            plt.title("Confusion Matrix (Normalized by Recall)")
            plt.show()
            results["MATRIX_RECALL"] = cm_recall

        elif m_upper == "MATRIX_PRED":
            cm_pred = confusion_matrix(target, pred, normalize='pred')
            ConfusionMatrixDisplay(cm_pred).plot(values_format='.2f', cmap='plasma')
            plt.title("Confusion Matrix (Normalized by Predictions)")
            plt.show()
            results["MATRIX_PRED"] = cm_pred

        elif m_upper.startswith("PRECISION_") or m_upper.startswith("RECALL_") or m_upper.startswith("F1_"):
            class_label_part = m.split("_", 1)[1]
            class_label, asimilation = find_class_label(unique_classes, class_label_part)
            if class_label is None:
                print(f"Class '{class_label_part}' not found. Skipping this metric.")
                continue
            if asimilation:
                print(f"Note: Class name '{class_label_part}' was assimilated to '{class_label}' for metric calculation.")
            
            if m_upper.startswith("PRECISION_"):
                prec_class = precision_score(target, pred, labels=[class_label], average=None)
                results[f"PRECISION_{class_label}"] = prec_class[0]
            elif m_upper.startswith("RECALL_"):
                rec_class = recall_score(target, pred, labels=[class_label], average=None)
                results[f"RECALL_{class_label}"] = rec_class[0]
            elif m_upper.startswith("F1_"):
                f1_class = f1_score(target, pred, labels=[class_label], average=None)
                results[f"F1_{class_label}"] = f1_class[0]

    return results

def eval_model(target, pred, problem_type=None, metric='ALL', average='macro'):
    """
    Evalúa las métricas de rendimiento para modelos de regresión o clasificación,
    llamando internamente a las funciones eval_reg o eval_clas según sea necesario.

    Esta función automatiza la evaluación de múltiples métricas de rendimiento,
    facilitando la comparación entre los valores predichos y los valores reales (target).
    Además, detecta automáticamente el tipo de problema (regresión o clasificación)
    basándose en el target si no se especifica explícitamente.

    Parámetros:
    - target: array-like, valores reales/target del conjunto de datos.
    - pred: array-like, valores predichos por el modelo.
    - problem_type: str (opcional), especifica el tipo de problema ('reg' para regresión o 'clas' para clasificación).
      Si es None, el tipo de problema se detecta automáticamente.
    - metric: str o list (opcional), especifica las métricas a evaluar. Si es 'all', se evalúan todas las métricas aplicables.
      Las métricas deben estar en mayúsculas o minúsculas o cualquier combinación de estas.
    - average: str (opcional, solo para clasificación), determina el método de promediado para métricas multiclase.
      Los valores válidos son 'macro', 'micro', 'samples', 'weighted' y 'binary'.

    Retorna:
    - Un diccionario con los nombres de las métricas evaluadas como claves y los resultados calculados como valores.
      Para métricas que se imprimen directamente en pantalla (por ejemplo, gráficas o reports),
      el valor correspondiente en el diccionario puede ser None o una representación textual si es aplicable.
    """

    # Detecta automáticamente el tipo de problema (regresión o clasificación) si no se proporciona
    if problem_type is None:
        problem_type = detect_problem_type(target)
    
    # Asegura que el argumento 'metric' sea una lista para un manejo uniforme
    if isinstance(metric, str):
        metric = [metric]
        
    # Convierte todas las métricas a mayúsculas para un manejo insensible a mayúsculas y minúsculas
    metric = [m.upper() for m in metric]
    
    # Inicializa el diccionario de resultados que almacenará los valores de las métricas calculadas
    results = {}

    # Si el tipo de problema es regresión, llama a la función eval_reg y pasa los parámetros necesarios
    if problem_type == "reg":
        results = eval_reg(target, pred, metric)
    # Si el tipo de problema es clasificación, llama a la función eval_clas y pasa los parámetros necesarios
    elif problem_type == "clas":
        results = eval_clas(target, pred, metric, average)
    # Maneja el caso de que se proporcione un tipo de problema desconocido
    else:
        raise ValueError(f"Unknown problem type: {problem_type}")

    # Retorna el diccionario de resultados con las métricas calculadas
    return results



#### SEGUNDA FUNCION: get_features_cat_num_classification

def get_features_cat_num_classification(df, target_col, pvalue_limit=0.05):
    """
    Esta función devuelve una lista de columnas categóricas o numéricas
    cuyo test Chi2 o ANOVA,respectivamente, respecto a la columna designada
    como target supera el test de hipótesis con una significación >= (1-pvalue_limit)

    Args:
        arg1 : Dataframe
        arg2 : Columna categórica o numérica discreta con baja cardinalidad seleccionada como target.
        arg3 : pvalue seleccionado como limite

    Retorns:
        Las columnas numericas o categoricas significativas

    Raises:
        TypeError: Si `df` no es un DataFrame de Pandas.
        ValueError: Si `target_col` no es una columna del DataFrame.
        TypeError: Si `p-value` no es un float en el rango de 0 a 1. 
    """
    # Comprobación de que df es un DataFrame(objeto,clase)
    if not isinstance(df, pd.DataFrame):
        print("Error: El primer argumento debe ser un DataFrame.")
        return None
    
    # Comprobación de que target_col es una columna en el DataFrame
    if target_col not in df.columns:
        print(f"Error: '{target_col}' no es una columna válida en el DataFrame.")
        return None
    
    # Comprobación de que pvalue es un valor válido
    if not (isinstance(pvalue_limit, float) and 0 < pvalue_limit < 1):
        print("Error: 'pvalue' debe ser un valor float en el rango (0, 1).")
        return None
    
    # Lista para almacenar las columnas significativas
    columnas_seleccionadas = []
    
    # Obtener los valores únicos de la columna target
    target_values = df[target_col].unique()
    
    # Comprobar si la variable target es categórica o numérica discreta
    for col in df.columns:
        if df[col].dtype == 'object' or df[col].nunique() <= 10: #  menos de 10 valores únicos es categórico
            categorica = True
        else:
            categorica = False
    
    # Iterar sobre las columnas numéricas y categóricas
        if categorica:
            # Realizar el test de Chi_cuadradado para variables categóricas
            tabla_contingencia = pd.crosstab(df[col], df[target_col])
            _, p_value, _, _ = chi2_contingency(tabla_contingencia)
            #no ponemos stadisttic ni tabla de libertad ya que no es necesario para esta funcion
        else:
            # Realizar el test ANOVA para variables numéricas
            groups = [group[col] for name, group in df.groupby(target_col)]
            _,p_value = f_oneway(*groups)
        
        # Comprobar si el p-value es menor o igual al límite
        if p_value <= (1 - pvalue_limit):
            columnas_seleccionadas.append(col)

        if target_col in columnas_seleccionadas:
            columnas_seleccionadas.remove(target_col)
            
         
    
    return columnas_seleccionadas


#TERCERA FUNCION: plot_features_cat_num_classification

def plot_features_cat_num_classification(df, target_col, columns=[], p_value=0.05, normalized=True, max_plots=5):
    """
  Crea un scatterplot o un countplot para cada variable numérica o categórica del DataFrame con respecto a la variable objetivo, respectivamente.

  Args:
      df: Un DataFrame.
      target_col: Columna categórica o numérica discreta con baja cardinalidad seleccionada como target.
      columns: columnas con significacion estadistica
      p-value= pvalue seleccionado como limite.
      normalized= True: (por defecto) calcula la frecuencia relativa para cada categoría en lugar del conteo bruto.
      max_plots: numero de plots maximo por figura 


  Returns:
      la función crea uno countplot para cada variable numérica o categórica del DataFrame con respecto al target,
      dibujando estos en figuras agrupdas segun el argumento max_plots establecido.
      En caso que la lista de columnas este vacia, se completara con una lista de las columnas del Dataframe.

  Raises:
      TypeError: Si `df` no es un DataFrame de Pandas.
      ValueError: Si `target_col` no es una columna del DataFrame.
      TypeError: Si `p-value` no es un float en el rango de 0 a 1. 
 
    """
    # Comprobación de que df es un DataFrame(objeto,clase)
    if not isinstance(df, pd.DataFrame):
        print("Error: El primer argumento debe ser un DataFrame.")
        return None
    
    # Comprobación de que target_col es una columna en el DataFrame
    if target_col not in df.columns:
        print(f"Error: '{target_col}' no es una columna válida en el DataFrame.")
        return None
    
    # Comprobación de que p_value es un valor válido
    if not (isinstance(p_value, float) and 0 < p_value < 1):
        print("Error: 'p_value' debe ser un valor float en el rango (0, 1).")
        return None
    
    # Si la lista de columnas está vacía, asignar todas las variables numéricas o categoricas del DataFrame
    if not columns:
        columns = df.select_dtypes(include=['number', "object"]).columns.tolist()
    
    # Realizar el test ANOVA o de chi-cuadrado y mantener solo las columnas que cumplen con el umbral de significancia
    # segun sea numérica o categórica
      # Lista para almacenar las columnas significativas
    col_significativas = []
    
    # Obtener los valores únicos de la columna target
    target_values = df[target_col].unique()
    
    # Comprobar si la variable target es categórica o numérica discreta
    for col in df.columns:
        if df[col].dtype == 'object' or df[col].nunique() <= 10: #  menos de 10 valores únicos es categórico
            categorica = True
        else:
            categorica = False
    
    # Iterar sobre las columnas numéricas y categóricas
        if categorica:
            # Realizar el test de Chi_cuadradado para variables categóricas
            tabla_contingencia = pd.crosstab(df[col], df[target_col])
            _, p_value, _, _ = chi2_contingency(tabla_contingencia)
            #no ponemos stadisttic ni tabla de libertad ya que no es necesario para esta funcion
        else:
            # Realizar el test ANOVA para variables numéricas
            groups = [group[col] for name, group in df.groupby(target_col)]
            _,p_value = f_oneway(*groups)
        
        # Comprobar si el p-value es menor o igual al límite
        if p_value <= (1 - p_value):
            col_significativas.append(col)   

    # Dividir las columnas en grupos de 'max_plots'  por figura
    column_groups = []
    # Itero sobre la lista 'col_significativas' con el tamaño marcado como pasos máximo deseado
    for i in range(0, len(col_significativas), max_plots):
    # Añade un subgrupo de 'max_plots' elementos a 'column_groups'
        column_groups.append(col_significativas[i:i+max_plots])
    
    # Generar plots para cada grupo de columnas
    for group in column_groups:
        if len(group) ==1: #comprobamos si solo hay 1 columna en el grupo
                fig, ax = plt.subplots(figsize=(4, 4))

        else:   
            num_plots = len(group)
            num_rows = (num_plots + 1) // 2
            num_cols = 2 if num_plots >1 else 1
            fig, axes = plt.subplots(num_rows, num_cols, figsize=(8, 4*num_rows))
        
         # Convertir axes en una lista de listas con un solo elemento 
        #unificando la extructura independiente de datos del numero de subplots
        axes = [axes] if num_rows == 1 else axes
        
        for i, col in enumerate(group):
            if num_plots == 1:  # Si hay solo un subplot
                ax = axes[i]
            else:
                #calculo el el numero de filas recorridas hasta el indica actual (i)
                row_idx = i // num_cols
                #calculo la posición dela figura en la fila
                col_idx = i % num_cols
                #contenedor de los subplots creados
                ax = axes[row_idx][col_idx]
               
            
            if df[col].dtype == 'object' or df[col].nunique() <= 10:
                sns.countplot(data=df, x=col, hue=target_col, ax=ax)
                ax.set_title(f'{col} vs {target_col}')
                ax.legend().remove()
            else:
                sns.scatterplot(data=df, x=col, y=target_col, hue=target_col, ax=ax, legend=ax)
                ax.set_title(f'{col} vs {target_col}')
                ax.legend().remove()
        
        plt.tight_layout()
        plt.show()
    
    return col_significativas


#CUARTA FUNCION: ### Funcion: 


def super_selector(dataset, target_col="", selectores='all', hard_voting=5):
    """
    Selecciona características de un dataframe de features según diferentes métodos de selección.

    Args:
        dataset (pandas.DataFrame): El dataframe de características.
        target_col (str): El nombre de la columna objetivo.
        selectores (dict): Un diccionario con métodos de selección como claves y sus parámetros como valores.
        hard_voting (list): Una lista de características para realizar hard voting o bien un entero que especifique el número de características que se quieren. 

    Returns:
        dict: Un diccionario que contiene las características seleccionadas por cada método de selección
              y el resultado del hard voting si se especifica.
    """
    # Verificar si target_col es válido
    if target_col and target_col in dataset.columns:  # Verifica si se proporcionó un nombre de columna objetivo y si existe en el dataframe.
        target = dataset[target_col]  # Guarda la columna objetivo en 'target'.
        features = dataset.drop(columns=[target_col])  # Elimina la columna objetivo del dataframe de características y guarda el resto en 'features'.
    else:
        print(f'"{target_col} " no es una columna del dataset, por favor proporciona una columna válida')
        return None

    if selectores == {} or selectores is None:
        # Filtrar columnas que no son el target, tienen una cardinalidad diferente del 99.99%, y no tienen un único valor.
        filtered_columns  = [col for col in dataset.columns if col != target_col and len(dataset[col].unique()) != 1 
                            and ((dataset[col].nunique() / len(dataset[col])) * 100) < 99.99]
        return filtered_columns
    
    if selectores == 'all':
        selectores = {
            "KBest": 5,  # Seleccionar las 10 mejores características
            "FromModel": [
                RandomForestClassifier(n_estimators=100, max_depth=None, min_samples_split=2), 
                5  # Número de características a seleccionar desde el modelo
            ],
            "RFE": [
                LogisticRegression(max_iter=2000, C=1.0, solver='lbfgs'), 
                5,  # Número de características a seleccionar
                1   # Paso para la eliminación de características
            ],
            "SFS": [
                RandomForestClassifier(n_estimators=100, max_depth=None, min_samples_split=2), 
                5  # Número de características a seleccionar con SFS
            ] 
        }

    result = {}  # Se inicializa un diccionario para almacenar el resultado de la selección de características.

    # Seleccionar características basadas en selectores

    for selector, params in selectores.items():  # Itera sobre cada método de selección y sus parámetros.
        if selector == "KBest":  # Si el método de selección es 'KBest' (Selección de las mejores características basadas en pruebas univariadas).
            k_best = SelectKBest(score_func=f_classif, k=params)  # Inicializa el selector de las mejores características.
            k_best.fit(features, target)  # Ajusta el selector a las características y el objetivo.
            selected_features = features.columns[k_best.get_support()]  # Obtiene las características seleccionadas.
            result[selector] = selected_features.tolist()  # Almacena las características seleccionadas en el resultado.

        elif selector == "FromModel":  # Si el método de selección es 'FromModel' (Selección de características basadas en un modelo externo).
            model, threshold = params  # Obtiene el modelo y el umbral de los parámetros.
            if isinstance(threshold, int):  # Comprueba si el umbral es un número entero.
                max_features = threshold  # Si es un número entero, se establece como el número máximo de características.
                threshold = -np.inf  # El umbral se establece en infinito negativo.
            else:
                max_features = None  # Si no es un número entero, el número máximo de características no se limita.
            selector_model = SelectFromModel(model, threshold=threshold, max_features=max_features)  # Inicializa el selector basado en el modelo.
            selector_model.fit(features, target)  # Ajusta el selector a las características y el objetivo.
            selected_features = features.columns[selector_model.get_support()]  # Obtiene las características seleccionadas.
            result[selector] = selected_features.tolist()  # Almacena las características seleccionadas en el resultado.

        elif selector == "RFE":  # Si el método de selección es 'RFE' (Eliminación recursiva de características).
            model, n_features, step = params  # Obtiene el modelo, el número de características y el paso de los parámetros.
            rfe = RFE(estimator=model, n_features_to_select=n_features, step=step)  # Inicializa el selector de RFE.
            rfe.fit(features, target)  # Ajusta el selector a las características y el objetivo.
            selected_features = features.columns[rfe.support_]  # Obtiene las características seleccionadas.
            result[selector] = selected_features.tolist()  # Almacena las características seleccionadas en el resultado.

        elif selector == "SFS":  # Si el método de selección es 'SFS' (Selección secuencial hacia adelante).
            model, n_features = params  # Obtiene el modelo y el número de características de los parámetros.
            sfs = SequentialFeatureSelector(model, k_features=n_features, forward=True, floating=False, scoring='accuracy', cv=StratifiedKFold(5))  # Inicializa el selector SFS.
            sfs.fit(features, target)  # Ajusta el selector a las características y el objetivo.
            selected_features = features.columns[list(sfs.k_feature_idx_)]  # Obtiene las características seleccionadas.
            result[selector] = selected_features.tolist()  # Almacena las características seleccionadas en el resultado.

    # Realizar hard voting
    
    # Determinar si hard_voting es una lista o un entero
    is_list = isinstance(hard_voting, list)

    # Unificar el tratamiento de all_features: suma de resultados + hard_voting si es una lista
    all_features = sum((result[key] for key in result), [])
    if is_list:
        all_features.extend(hard_voting)

    if all_features:
        # Contar la frecuencia de cada característica en las listas combinadas
        voting_counts = Counter(all_features)
        # Ordenar las características por frecuencia en orden descendente
        sorted_voting = [feature for feature, _ in voting_counts.most_common()]

        # Seleccionar características basadas en hard_voting: lista = longitud de hard_voting; entero = top N características
        n_features = len(hard_voting) if is_list else hard_voting
        hard_voting_result = sorted_voting[:n_features]

        result['hard_voting'] = hard_voting_result
    else:
        print('No se han seleccionado características, revise los parámetros de entrada')

    return result  # Devuelve el resultado que contiene las características seleccionadas por cada método de selección y el resultado del hard voting.