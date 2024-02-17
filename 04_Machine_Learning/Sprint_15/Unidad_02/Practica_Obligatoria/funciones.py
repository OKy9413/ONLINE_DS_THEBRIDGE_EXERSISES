import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np


def pinta_distribucion_categoricas(df, columnas_categoricas, relativa=False, mostrar_valores=False):
    num_columnas = len(columnas_categoricas)
    num_filas = (num_columnas // 2) + (num_columnas % 2)

    fig, axes = plt.subplots(num_filas, 2, figsize=(15, 5 * num_filas))
    axes = axes.flatten() 

    for i, col in enumerate(columnas_categoricas):
        ax = axes[i]
        if relativa:
            total = df[col].value_counts().sum()
            serie = df[col].value_counts().apply(lambda x: x / total)
            sns.barplot(x=serie.index, y=serie, ax=ax, palette='viridis', hue = serie.index, legend = False)
            ax.set_ylabel('Frecuencia Relativa')
        else:
            serie = df[col].value_counts()
            sns.barplot(x=serie.index, y=serie, ax=ax, palette='viridis', hue = serie.index, legend = False)
            ax.set_ylabel('Frecuencia')

        ax.set_title(f'Distribución de {col}')
        ax.set_xlabel('')
        ax.tick_params(axis='x', rotation=45)

        if mostrar_valores:
            for p in ax.patches:
                height = p.get_height()
                ax.annotate(f'{height:.2f}', (p.get_x() + p.get_width() / 2., height), 
                            ha='center', va='center', xytext=(0, 9), textcoords='offset points')

    for j in range(i + 1, num_filas * 2):
        axes[j].axis('off')

    plt.tight_layout()
    plt.show()


def plot_categorical_relationship_fin(df, cat_col1, cat_col2, relative_freq=False, show_values=False, size_group = 5):
    # Prepara los datos
    count_data = df.groupby([cat_col1, cat_col2]).size().reset_index(name='count')
    total_counts = df[cat_col1].value_counts()
    
    # Convierte a frecuencias relativas si se solicita
    if relative_freq:
        count_data['count'] = count_data.apply(lambda x: x['count'] / total_counts[x[cat_col1]], axis=1)

    # Si hay más de size_group categorías en cat_col1, las divide en grupos de size_group
    unique_categories = df[cat_col1].unique()
    if len(unique_categories) > size_group:
        num_plots = int(np.ceil(len(unique_categories) / size_group))

        for i in range(num_plots):
            # Selecciona un subconjunto de categorías para cada gráfico
            categories_subset = unique_categories[i * size_group:(i + 1) * size_group]
            data_subset = count_data[count_data[cat_col1].isin(categories_subset)]

            # Crea el gráfico
            plt.figure(figsize=(10, 6))
            ax = sns.barplot(x=cat_col1, y='count', hue=cat_col2, data=data_subset, order=categories_subset)

            # Añade títulos y etiquetas
            plt.title(f'Relación entre {cat_col1} y {cat_col2} - Grupo {i + 1}')
            plt.xlabel(cat_col1)
            plt.ylabel('Frecuencia' if relative_freq else 'Conteo')
            plt.xticks(rotation=45)

            # Mostrar valores en el gráfico
            if show_values:
                for p in ax.patches:
                    ax.annotate(f'{p.get_height():.2f}', (p.get_x() + p.get_width() / 2., p.get_height()),
                                ha='center', va='center', fontsize=10, color='black', xytext=(0, size_group),
                                textcoords='offset points')

            # Muestra el gráfico
            plt.show()
    else:
        # Crea el gráfico para menos de size_group categorías
        plt.figure(figsize=(10, 6))
        ax = sns.barplot(x=cat_col1, y='count', hue=cat_col2, data=count_data)

        # Añade títulos y etiquetas
        plt.title(f'Relación entre {cat_col1} y {cat_col2}')
        plt.xlabel(cat_col1)
        plt.ylabel('Frecuencia' if relative_freq else 'Conteo')
        plt.xticks(rotation=45)

        # Mostrar valores en el gráfico
        if show_values:
            for p in ax.patches:
                ax.annotate(f'{p.get_height():.2f}', (p.get_x() + p.get_width() / 2., p.get_height()),
                            ha='center', va='center', fontsize=10, color='black', xytext=(0, size_group),
                            textcoords='offset points')

        # Muestra el gráfico
        plt.show()


def plot_categorical_numerical_relationship(df, categorical_col, numerical_col, show_values=False, measure='mean'):
    # Calcula la medida de tendencia central (mean o median)
    if measure == 'median':
        grouped_data = df.groupby(categorical_col)[numerical_col].median()
    else:
        # Por defecto, usa la media
        grouped_data = df.groupby(categorical_col)[numerical_col].mean()

    # Ordena los valores
    grouped_data = grouped_data.sort_values(ascending=False)

    # Si hay más de 5 categorías, las divide en grupos de 5
    if grouped_data.shape[0] > 5:
        unique_categories = grouped_data.index.unique()
        num_plots = int(np.ceil(len(unique_categories) / 5))

        for i in range(num_plots):
            # Selecciona un subconjunto de categorías para cada gráfico
            categories_subset = unique_categories[i * 5:(i + 1) * 5]
            data_subset = grouped_data.loc[categories_subset]

            # Crea el gráfico
            plt.figure(figsize=(10, 6))
            ax = sns.barplot(x=data_subset.index, y=data_subset.values)

            # Añade títulos y etiquetas
            plt.title(f'Relación entre {categorical_col} y {numerical_col} - Grupo {i + 1}')
            plt.xlabel(categorical_col)
            plt.ylabel(f'{measure.capitalize()} de {numerical_col}')
            plt.xticks(rotation=45)

            # Mostrar valores en el gráfico
            if show_values:
                for p in ax.patches:
                    ax.annotate(f'{p.get_height():.2f}', (p.get_x() + p.get_width() / 2., p.get_height()),
                                ha='center', va='center', fontsize=10, color='black', xytext=(0, 5),
                                textcoords='offset points')

            # Muestra el gráfico
            plt.show()
    else:
        # Crea el gráfico para menos de 5 categorías
        plt.figure(figsize=(10, 6))
        ax = sns.barplot(x=grouped_data.index, y=grouped_data.values)

        # Añade títulos y etiquetas
        plt.title(f'Relación entre {categorical_col} y {numerical_col}')
        plt.xlabel(categorical_col)
        plt.ylabel(f'{measure.capitalize()} de {numerical_col}')
        plt.xticks(rotation=45)

        # Mostrar valores en el gráfico
        if show_values:
            for p in ax.patches:
                ax.annotate(f'{p.get_height():.2f}', (p.get_x() + p.get_width() / 2., p.get_height()),
                            ha='center', va='center', fontsize=10, color='black', xytext=(0, 5),
                            textcoords='offset points')

        # Muestra el gráfico
        plt.show()


def plot_combined_graphs(df, columns, whisker_width=1.5, bins = None):
    num_cols = len(columns)
    if num_cols:
        
        fig, axes = plt.subplots(num_cols, 2, figsize=(12, 5 * num_cols))
        print(axes.shape)

        for i, column in enumerate(columns):
            if df[column].dtype in ['int64', 'float64']:
                # Histograma y KDE
                sns.histplot(df[column], kde=True, ax=axes[i,0] if num_cols > 1 else axes[0], bins= "auto" if not bins else bins)
                if num_cols > 1:
                    axes[i,0].set_title(f'Histograma y KDE de {column}')
                else:
                    axes[0].set_title(f'Histograma y KDE de {column}')

                # Boxplot
                sns.boxplot(x=df[column], ax=axes[i,1] if num_cols > 1 else axes[1], whis=whisker_width)
                if num_cols > 1:
                    axes[i,1].set_title(f'Boxplot de {column}')
                else:
                    axes[1].set_title(f'Boxplot de {column}')

        plt.tight_layout()
        plt.show()

def plot_grouped_boxplots(df, cat_col, num_col):
    unique_cats = df[cat_col].unique()
    num_cats = len(unique_cats)
    group_size = 5

    for i in range(0, num_cats, group_size):
        subset_cats = unique_cats[i:i+group_size]
        subset_df = df[df[cat_col].isin(subset_cats)]
        
        plt.figure(figsize=(10, 6))
        sns.boxplot(x=cat_col, y=num_col, data=subset_df)
        plt.title(f'Boxplots of {num_col} for {cat_col} (Group {i//group_size + 1})')
        plt.xticks(rotation=45)
        plt.show()



def plot_grouped_histograms(df, cat_col, num_col, group_size):
    unique_cats = df[cat_col].unique()
    num_cats = len(unique_cats)

    for i in range(0, num_cats, group_size):
        subset_cats = unique_cats[i:i+group_size]
        subset_df = df[df[cat_col].isin(subset_cats)]
        
        plt.figure(figsize=(10, 6))
        for cat in subset_cats:
            sns.histplot(subset_df[subset_df[cat_col] == cat][num_col], kde=True, label=str(cat))
        
        plt.title(f'Histograms of {num_col} for {cat_col} (Group {i//group_size + 1})')
        plt.xlabel(num_col)
        plt.ylabel('Frequency')
        plt.legend()
        plt.show()



def grafico_dispersion_con_correlacion(df, columna_x, columna_y, tamano_puntos=50, mostrar_correlacion=False):
    """
    Crea un diagrama de dispersión entre dos columnas y opcionalmente muestra la correlación.

    Args:
    df (pandas.DataFrame): DataFrame que contiene los datos.
    columna_x (str): Nombre de la columna para el eje X.
    columna_y (str): Nombre de la columna para el eje Y.
    tamano_puntos (int, opcional): Tamaño de los puntos en el gráfico. Por defecto es 50.
    mostrar_correlacion (bool, opcional): Si es True, muestra la correlación en el gráfico. Por defecto es False.
    """

    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x=columna_x, y=columna_y, s=tamano_puntos)

    if mostrar_correlacion:
        correlacion = df[[columna_x, columna_y]].corr().iloc[0, 1]
        plt.title(f'Diagrama de Dispersión con Correlación: {correlacion:.2f}')
    else:
        plt.title('Diagrama de Dispersión')

    plt.xlabel(columna_x)
    plt.ylabel(columna_y)
    plt.grid(True)
    plt.show()


def bubble_plot(df, col_x, col_y, col_size, scale = 1000):
    """
    Crea un scatter plot usando dos columnas para los ejes X e Y,
    y una tercera columna para determinar el tamaño de los puntos.

    Args:
    df (pd.DataFrame): DataFrame de pandas.
    col_x (str): Nombre de la columna para el eje X.
    col_y (str): Nombre de la columna para el eje Y.
    col_size (str): Nombre de la columna para determinar el tamaño de los puntos.
    """

    # Asegúrate de que los valores de tamaño sean positivos
    sizes = (df[col_size] - df[col_size].min() + 1)/scale

    plt.scatter(df[col_x], df[col_y], s=sizes)
    plt.xlabel(col_x)
    plt.ylabel(col_y)
    plt.title(f'Burbujas de {col_x} vs {col_y} con Tamaño basado en {col_size}')
    plt.show()


def plot_multiple_boxplots(df, columns, dim_matriz_visual = 2):
    num_cols = len(columns)
    num_rows = num_cols // dim_matriz_visual + num_cols % dim_matriz_visual
    fig, axes = plt.subplots(num_rows, dim_matriz_visual, figsize=(12, 6 * num_rows))
    axes = axes.flatten()

    for i, column in enumerate(columns):
        if df[column].dtype in ['int64', 'float64']:
            sns.boxplot(data=df, x=column, ax=axes[i])
            axes[i].set_title(column)

    # Ocultar ejes vacíos
    for j in range(i+1, num_rows * 2):
        axes[j].axis('off')

    plt.tight_layout()
    plt.show()


def plot_histo_dens(df, columns, bins=None):
    num_cols = len(columns)
    num_rows = num_cols // 2 + num_cols % 2
    fig, axes = plt.subplots(num_rows, 2, figsize=(12, 6 * num_rows))
    axes = axes.flatten()

    for i, column in enumerate(columns):
        if df[column].dtype in ['int64', 'float64']:
            if bins:
                sns.histplot(df[column], kde=True, ax=axes[i], bins=bins)
            else:
                sns.histplot(df[column], kde=True, ax=axes[i])
            axes[i].set_title(f'Histograma y KDE de {column}')

    # Ocultar ejes vacíos
    for j in range(i + 1, num_rows * 2):
        axes[j].axis('off')

    plt.tight_layout()
    plt.show()

# Ejemplo de uso:
# plot_histograms_with_density(df, ['columna1', 'columna2', 'columna3'], bins=20)


def ver_corr_col(c1, c2, df):
    '''
    Esta función comprueba que existe una correlación directa entre dos columnas. Si lo es te muestra los valores únicos de cada una.

    Args:
    Hay que pasarle las dos columnas como str, y el nombre del df.

    Return:
    Devuelve una tupla. True y los valores únicos de las columnas. False y mensajes sobre por qué no cuadra.
    '''
    a = df[df[c1].notna()][c1].unique().tolist()       # Sacamos los valores únicos de cada una de las columnas
    b = df[df[c2].notna()][c2].unique().tolist()
    dic = {}
    if len(a) == len(b):
        for i in range(len(a)):
            dic[a[i]] = b[i]             
        for index, row in df.iterrows():        # Verificar que todas las filas coincidan con el diccionario
            valor_c1 = row[c1]
            valor_c2 = row[c2]

            if valor_c1 in dic and dic[valor_c1] != valor_c2:       # Si hay discrepancia, devuelve false
                return (False,'Hay discrepancias de correspondencia')   # El sigiente paso sería ver en qué porcentaje es el que varía. Pero haría falta rehacer la creación del dic
                break                   
    
        return (True,a,b)       # Devuelve True si todos lo valores corresponden. Así como las dos listas de valores únicos.
    else:
        return (False,'Hay diferente número de valores únicos')           # Longitudes de listas diferentes, por lo que no se corresponden



def drop_col_corr(df, drop_col = True):
    """
    Función para comprobar la correspondencia entre todas las columnas de un dataframe y, en caso de que exista, eliminar la segunda columna.
    Modifica el dataframe directamente y devuelve un diccionario con las columnas dropeadas.

    Args:
        df: Dataframe a comprobar.
        drop_col: Bool. Por defecto es True. Poner false si no queremos que nos elimine las colunmas directamente.

    Returns:
        Diccionario con las columnas dropeadas.
    """
    # Recorremos todas las columnas
    correspondencias = {}
    columnas_disp = df.columns.tolist().copy()
    for c1 in columnas_disp:
        for c2 in columnas_disp:
            if c1 != c2:
                res = ver_corr_col(c1, c2, df)
                if res[0]:
                    correspondencias[c1] = c2
                    # Eliminamos la columna de los iterables
                    columnas_disp.remove(c2)
                    
    # Eliminamos las columnas del dataframe
    if drop_col is True:
        for c1 in correspondencias:
            df.drop(columns=[correspondencias[c1]], inplace=True)
    return correspondencias


def drop_selcol_corr(df, drop_col=True):
    """
    Función para comprobar la correspondencia entre todas las columnas de un dataframe y, en caso de que exista, eliminar una columna a tu elección.
    Modifica el dataframe directamente y devuelve un diccionario con las columnas dropeadas.

    Args:
        df: Dataframe a comprobar.
        drop_col: Bool. Por defecto es True. Poner False si no queremos que nos elimine las columnas directamente.

    Returns:
        Diccionario con las columnas dropeadas.
    """
    correspondencias = {}
    columnas_disp = df.columns.tolist().copy()
    opciones = {}  # Diccionario para almacenar las opciones del usuario
    
    # Buscar correspondencias entre columnas
    for c1 in columnas_disp:
        for c2 in columnas_disp:
            if c1 != c2:
                res = ver_corr_col(c1, c2, df)
                if res[0] and (c1, c2) not in opciones.values() and (c2, c1) not in opciones.values():
                    opciones[len(opciones) + 1] = (c1, c2)
    
    # Mostrar las opciones al usuario y solicitar las elecciones
    if opciones:
        print("Parejas de columnas con correspondencia:")
        for idx, (col1, col2) in opciones.items():
            print(f"{idx}: {col1} - {col2}")
            eleccion = int(input(f"Ingrese 1 para eliminar '{col2}' o 2 para eliminar '{col1}': "))
            if eleccion == 1:
                correspondencias[col1] = col2
            elif eleccion == 2:
                correspondencias[col2] = col1
            else:
                print("Opción inválida")
        
        # Eliminar las columnas elegidas por el usuario
        if drop_col is True:
            for col_a_eliminar, col_b_eliminar in correspondencias.items():
                df.drop(columns=[col_b_eliminar], inplace=True)  # Eliminar la segunda columna de la pareja
                print(f"Se eliminó la columna '{col_b_eliminar}' de la pareja '{col_a_eliminar} - {col_b_eliminar}'")
    else:
        print("No se encontraron correspondencias entre las columnas")
    
    return correspondencias


def fillna_media_referida(fillcol, refcol, df):
    '''
    La función sustituye los valores nulos de una columna en función de la media de sus valores no nulos 
    en una seleción de las filas en referencia a otra columna (esperablemente categórica).
    Primero se ingresa la columna a rellenar y luego la columna de referencia. Las dos como str.
    Colocar el nombre del df en tercer lugar.

    Return:
    No devuelve nada, solo rellena el df.
    '''
    sel = df[df[refcol].notna()][refcol].unique().tolist()              # Aquí obtenemos los valores unicos de la columna para poder hacer la selección
    for i in sel:
        val = (df.loc[df[refcol] == i, fillcol].mean()).round(2)        # Aquí calculamos la media de la selección de valores que no son nulos
        df.loc[df[refcol] == i, fillcol] = df.loc[df[refcol] == i, fillcol].fillna(val)    # Y aquí rellenamos los nulos de esa selección con esa media
        # Importante!!! el implace aquí no sirve, porque solo me modifica la varible local del df, no la global, para hacerlo necesito renombrar la selección que modifico!!!


def ver_corr_NaN(df, c1, c2):
    '''
    Esta función muestra la correspondencia de una columna con los nulos de otra.

    Args:
    Un df, y dos columnas pasadas como str.

    Return:
    no devuelve nada, solo hace print.
    '''
    sel = df[df.embarked.notna()][c1].unique().tolist()
    print(sel)
    for i in sel:
        part = ((df[c1] == i).mean() * 100).round(2)
        cond = (df[df[c1] == i][c2].isna().mean() * 100).round(2)
        print(f'El {part}% de los pasajeros tienen {c1} = {i}, y de esos, el {cond}% tienen {c2} como NaN', end = '\n')


def crea_df_std(df, row_names='all', col_names=['name', 'type', 'prio', 'card', 'card%', 'NaN', 'Unknown', '%_NaN', 'Category']):
    """
    Crea un DataFrame de resumen (df_std) para analizar la estructura de datos de un DataFrame dado (df).
    Proporciona información sobre el tipo de datos, la cardinalidad, el número de valores NaN y Unknown,
    el porcentaje de valores NaN y Unknown (%_NaN), y categoriza cada columna en 'Binaria', 'Categórica', 
    'Numérica Discreta' o 'Numérica Continua' basado en su cardinalidad y tipo de datos.

    Args:
        df (DataFrame): DataFrame de pandas a analizar.
        row_names (list, optional): Lista de nombres de columnas a incluir en el análisis. 'all' analiza todas las columnas.
        col_names (list, optional): Nombres de las columnas en el DataFrame de resumen.

    Returns:
        DataFrame: Un DataFrame de resumen con las columnas especificadas en col_names.
    """

    if row_names == 'all':
        row_names = df.columns

    row_types = []
    row_prio = []
    card = []
    card_per = []
    nan_counts = []
    unknown_counts = []
    null_per = []
    categories = []

    for col_name in row_names:
        col_type = df[col_name].dtype
        unique_values = df[col_name].nunique()
        percent_unique = round((unique_values / len(df) * 100), 2)
        nan_count = df[col_name].isna().sum()
        # Cuenta 'Unknown' en diferentes capitalizaciones
        unknown_count = df[col_name].apply(lambda x: str(x).lower() == 'unknown').sum()
        total_missing = nan_count + unknown_count
        null_percent = round((total_missing / len(df) * 100), 2)
        
        if unique_values == 2:
            category = 'Binaria'
        elif pd.api.types.is_numeric_dtype(col_type) and (unique_values > 7 and unique_values <= 20 or (percent_unique >= 5 and percent_unique <= 20)):
            category = 'Numérica Discreta'
        elif unique_values <= 10 and percent_unique > 1 or unique_values <= 7:
            category = 'Categórica'
        else:  # Para el resto de casos, consideramos la columna como Numérica Continua
            category = 'Numérica Continua'
        
        row_types.append(col_type)
        row_prio.append(3)  # Valor predeterminado, considerar hacerlo configurable o explicar su significado
        card.append(unique_values)
        card_per.append(percent_unique)
        nan_counts.append(nan_count)
        unknown_counts.append(unknown_count)
        null_per.append(null_percent)
        categories.append(category)
    
    df_std = pd.DataFrame(list(zip(row_names, row_types, row_prio, card, card_per, nan_counts, unknown_counts, null_per, categories)), columns=col_names)
    return df_std

def analyze_null_values_grouped(df, nul_cols='all', columns='all'):
    """
    Analiza los valores nulos en un DataFrame agrupados por columnas especificadas y calcula varias métricas:
    - '%_values': El porcentaje de cada valor único (o rango para columnas numéricas con más de 10 valores únicos)
      dentro de los nulos de una columna específica respecto al total de nulos en esa columna.
    - 'All': El número total de ocurrencias de cada valor único o rango en toda la columna analizada, no solo entre los nulos.
    - '%_All': El porcentaje que representa 'All' sobre el total de filas en el DataFrame.
    - 'Variation': La diferencia porcentual entre '%_values' y '%_All', mostrando cómo varía la distribución de valores nulos
      respecto a la distribución general en la columna.
    
    Args:
        df (pd.DataFrame): DataFrame de Pandas a analizar.
        nul_cols (list or str, optional): Columnas en las cuales buscar nulos. 'all' para todas las columnas.
        columns (list or str, optional): Columnas objetivo para mostrar el reparto de valores. 'all' para todas las columnas.
    
    Returns:
        list: Una lista de DataFrames, cada uno correspondiendo a una columna en `columns`, con las métricas calculadas.
    """
    if nul_cols == 'all':
        nul_cols = df.columns
    elif isinstance(nul_cols, str):
        nul_cols = [nul_cols]

    if columns == 'all':
        columns = df.columns
    elif isinstance(columns, str):
        columns = [columns]
    
    final_dfs = {}
    
    for colu in columns:
        all_dfs = []
        # Preparar 'temp_value' para todo el DataFrame antes del bucle de columnas
        if df[colu].dtype.kind in 'iuf' and df[colu].nunique() > 10:
            bins = np.linspace(df[colu].min(), df[colu].max(), 11)
            labels = [f"{bins[i]}-{bins[i+1]}" for i in range(10)]
            df['temp_value'] = pd.cut(df[colu], bins=bins, labels=labels, include_lowest=True)
        else:
            df['temp_value'] = df[colu].astype(str)
        
        # Calcular 'All' para todo el DataFrame
        all_counts = df['temp_value'].value_counts().reset_index()
        all_counts.columns = ['Value', 'All']
        all_counts['%_All'] = (all_counts['All'] / len(df)) * 100
        
        for col in nul_cols:
            mask = df[col].apply(lambda x: pd.isna(x) or str(x).lower() in ['unknown'])
            if not mask.any():
                continue
            
            # Agrupar por 'temp_value' dentro de los nulos y contar
            n_values_df = df.loc[mask, 'temp_value'].value_counts().reset_index()
            n_values_df.columns = ['Value', 'n_values']
            n_values_df['Columns'] = col
            
            # Calcular '%_values' como el porcentaje de nulos en 'col' para cada valor único o rango en 'colu'
            total_nulos_col = mask.sum()
            n_values_df['%_values'] = (n_values_df['n_values'] / total_nulos_col) * 100
            
            # Combinar con 'All' y '%_All'
            merged_df = pd.merge(n_values_df, all_counts, on='Value')
            merged_df['Variation'] = merged_df['%_values'] - merged_df['%_All']
            
            all_dfs.append(merged_df)
        
        if all_dfs:
            final_df = pd.concat(all_dfs, ignore_index=True)
            final_dfs[colu] = final_df[['Columns', 'Value', 'n_values', '%_values', 'All', '%_All', 'Variation']]
    
    # Mostrar y devolver los DataFrames finales
    for colu, df_colu in final_dfs.items():
        print(f"Análisis agrupado por '{colu}':")
        display(df_colu)
    
    return list(final_dfs.values())


def find_non_null_indices(dataframe, col1, col2):
    """
    Encuentra índices de filas donde el valor en col1 es nulo y el valor en col2 no es nulo.

    :param dataframe: DataFrame en el que buscar.
    :param col1: Nombre de la primera columna para comprobar si es nulo.
    :param col2: Nombre de la segunda columna para comprobar si no es nulo.
    :return: Lista de índices de filas que cumplen la condición.
    """
    indices = []
    for index, row in dataframe.iterrows():
        if pd.isna(row[col1]) and not pd.isna(row[col2]):
            indices.append(index)
    return indices


def unique_values_in_rows(df, indices, columns = None):
    """
    Muestra los valores únicos en las filas especificadas para las columnas dadas.

    :param df: DataFrame de Pandas.
    :param indices: Lista de índices de las filas.
    :param columns: Lista de columnas para mostrar valores únicos.
    :return: Un diccionario con los valores únicos para cada columna.
    """
    if columns is None:
        columns = df.columns
    unique_values = {}
    for column in columns:
        # Filtrar por los índices dados y luego extraer valores únicos para la columna
        unique_values[column] = df.loc[indices, column].unique()
    return unique_values


def fillna_valor_mas_cercano(df, fillcol, fecha_col, provincia_col):
    '''
    Rellena los valores nulos de una columna en función del valor más cercano en días
    para una provincia dada.
    
    :param df: DataFrame
    :param fillcol: Nombre de la columna a rellenar.
    :param fecha_col: Nombre de la columna de fecha.
    :param provincia_col: Nombre de la columna de provincia.
    '''
    # Asegurarse de que la columna de fecha sea de tipo datetime
    df[fecha_col] = pd.to_datetime(df[fecha_col])

    # Obtener las provincias únicas
    # Obtener los índices de las filas donde 'fillcol' es nulo
    indices_nulos = df[df[fillcol].isna()].index.tolist()

    # Usar la función unique_values_in_rows para obtener los valores únicos de 'provincia_col'
    valores_unicos = unique_values_in_rows(df, indices_nulos, [provincia_col])

    # Extraer las provincias únicas desde el diccionario
    provincias = valores_unicos[provincia_col]


    for provincia in provincias:
        # Filtrar el DataFrame por provincia
        df_prov = df[df[provincia_col] == provincia].sort_values(fecha_col)

        for index, row in df_prov.iterrows():
            if pd.isna(row[fillcol]):
                fecha = row[fecha_col]
                dias = 1
                valor_encontrado = False

                while not valor_encontrado:
                    # Buscar en días anteriores y posteriores
                    dia_anterior = fecha - pd.Timedelta(days=dias)
                    dia_posterior = fecha + pd.Timedelta(days=dias)

                    # Valores en días adyacentes
                    valor_anterior = df_prov[df_prov[fecha_col] == dia_anterior][fillcol].dropna()
                    valor_posterior = df_prov[df_prov[fecha_col] == dia_posterior][fillcol].dropna()

                    if not valor_anterior.empty:
                        df.at[index, fillcol] = valor_anterior.iloc[0]
                        valor_encontrado = True
                    elif not valor_posterior.empty:
                        df.at[index, fillcol] = valor_posterior.iloc[0]
                        valor_encontrado = True

                    dias += 1  # Incrementar el rango de búsqueda


