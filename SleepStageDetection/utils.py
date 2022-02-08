from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
import matplotlib.pyplot as plt
import matplotlib as mt
import pandas as pd
import numpy as np

def _optimize_floats(df: pd.DataFrame) -> pd.DataFrame:
    floats = df.select_dtypes(include=['float64']).columns.tolist()
    df[floats] = df[floats].apply(pd.to_numeric, downcast='float')
    return df

def _optimize_ints(df: pd.DataFrame) -> pd.DataFrame:
    ints = df.select_dtypes(include=['int64']).columns.tolist()
    df[ints] = df[ints].apply(pd.to_numeric, downcast='integer')
    return df

def optimize(df: pd.DataFrame) -> pd.DataFrame:
    return _optimize_floats(_optimize_ints(df))

def merge_dicts(*args):
    aux = {}
    for item in args:
        aux.update(item)
    return aux

def get_class_count(data:pd.DataFrame, column_class:str):
    return data[column_class].value_counts()

def barplot_class(ds:pd.Series):
    ax = ds.plot.bar(x='classes', y='quantidade', rot=0)
    plt.show()

def balanceamento_por_classe_oversampling(data:pd.DataFrame, class_column:str, plot:bool=False):
    """
    Consiste em reduzir de forma aleatória os exemplos da classe majoritária. 
    """
    resources = data.columns.tolist()
    resources.pop() #remover coluna da classe (valor y)
            
    under = RandomOverSampler()
    X, y = under.fit_resample(data[resources], data[class_column])

    if(plot == True):
        ds = y.value_counts()
        barplot_class(ds)

    return X, y

def balanceamento_por_classe_undersampling(data:pd.DataFrame, resources:list, class_column:str, plot:bool=False):
    """
    Under-sample the majority class(es) by randomly picking samples with or without replacement.. 
    """     
    under = RandomUnderSampler()
    X, y = under.fit_resample(data[resources], data[class_column])

    if(plot == True):
        ds = y.value_counts()
        barplot_class(ds)

    return X, y

def preparing_for_binary_classification(data:pd.DataFrame, class_column:str, desired_class:int):
    aux = data.copy()
    aux[class_column] = aux[class_column].apply(lambda x: 1 if x == desired_class else 0)
    return aux

def _preparing_for_binary_classification(data:pd.DataFrame):
    for i in range(0, 5):
        data[f"bin_{i}"] = data["stage"].apply(lambda x: 1 if x == i else 0)