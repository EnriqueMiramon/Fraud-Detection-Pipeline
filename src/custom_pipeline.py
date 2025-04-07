import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer, StandardScaler
from sklearn.compose import ColumnTransformer

#############################################
#           Funciones y Transformadores     #
#############################################

# Definimos el transformador para eliminar columnas
class DropColumns(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        # Columnas con alta cardinalidad
        high_cardinality_cols = ['P_emaildomain', 'R_emaildomain', 'id_30', 'id_31', 'id_33', 'DeviceInfo']

        # Columnas con muchos valores nulos (umbral 50%)
        null_cols = ['dist1', 'dist2', 'D2', 'D5', 'D6', 'D7', 'D8', 'D9', 'D11', 'D12', 'D13', 'D14', 'M1', 'M2', 'M3', 'M5', 'M7',
                    'M8', 'M9', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10', 'V11', 'V138', 'V139', 'V140', 'V141',
                    'V142', 'V143', 'V144', 'V145', 'V146', 'V147', 'V148', 'V149', 'V150', 'V151', 'V152', 'V153', 'V154', 'V155',
                    'V156', 'V157', 'V158', 'V159', 'V160', 'V161', 'V162', 'V163', 'V164', 'V165', 'V166', 'V167', 'V168', 'V169',
                    'V170', 'V171', 'V172', 'V173', 'V174', 'V175', 'V176', 'V177', 'V178', 'V179', 'V180', 'V181', 'V182', 'V183',
                    'V184', 'V185', 'V186', 'V187', 'V188', 'V189', 'V190', 'V191', 'V192', 'V193', 'V194', 'V195', 'V196', 'V197',
                    'V198', 'V199', 'V200', 'V201', 'V202', 'V203', 'V204', 'V205', 'V206', 'V207', 'V208', 'V209', 'V210', 'V211',
                    'V212', 'V213', 'V214', 'V215', 'V216', 'V217', 'V218', 'V219', 'V220', 'V221', 'V222', 'V223', 'V224', 'V225',
                    'V226', 'V227', 'V228', 'V229', 'V230', 'V231', 'V232', 'V233', 'V234', 'V235', 'V236', 'V237', 'V238', 'V239',
                    'V240', 'V241', 'V242', 'V243', 'V244', 'V245', 'V246', 'V247', 'V248', 'V249', 'V250', 'V251', 'V252', 'V253',
                    'V254', 'V255', 'V256', 'V257', 'V258', 'V259', 'V260', 'V261', 'V262', 'V263', 'V264', 'V265', 'V266', 'V267',
                    'V268', 'V269', 'V270', 'V271', 'V272', 'V273', 'V274', 'V275', 'V276', 'V277', 'V278', 'V322', 'V323', 'V324',
                    'V325', 'V326', 'V327', 'V328', 'V329', 'V330', 'V331', 'V332', 'V333', 'V334', 'V335', 'V336', 'V337', 'V338',
                    'V339', 'id_01', 'id_02', 'id_03', 'id_04', 'id_05', 'id_06', 'id_07', 'id_08', 'id_09', 'id_10', 'id_11', 'id_12',
                    'id_13', 'id_14', 'id_15', 'id_16', 'id_17', 'id_18', 'id_19', 'id_20', 'id_21', 'id_22', 'id_23', 'id_24', 'id_25',
                    'id_26', 'id_27', 'id_28', 'id_29', 'id_32', 'id_34', 'id_35', 'id_36', 'id_37', 'id_38', 'DeviceType']

        # Columnas constantes
        constant_cols = ['V305']

        # Columnas altamente correlacionadas
        corr_cols = {'V62', 'V84', 'C2', 'C11', 'V302', 'C14', 'V45', 'V301', 'V65', 'V280', 'V279', 'V299', 'V308', 'V76', 'C12',
                    'V309', 'V42', 'C6', 'C9', 'V64', 'V73', 'V33', 'V134', 'V137', 'V93', 'V94', 'V28', 'V90', 'V38', 'V125', 'V136',
                    'V119', 'V307', 'V30', 'V79', 'V60', 'V24', 'V113', 'V128', 'V48', 'V34', 'V22', 'V70', 'V91', 'V133', 'V291',
                    'V40', 'V97', 'V285', 'V292', 'C4', 'V36', 'V49', 'V105', 'V87', 'V294', 'V32', 'V284', 'V109', 'V306', 'V289',
                    'V297', 'V106', 'V124', 'V88', 'V102', 'V89', 'V31', 'V316', 'V16', 'V63', 'V13', 'V295', 'V101', 'V83', 'V311',
                    'C10', 'V131', 'V72', 'V43', 'V96', 'C13', 'V20', 'V69', 'V112', 'V21', 'V100', 'V318', 'V54', 'V111', 'V123',
                    'V317', 'V303', 'V57', 'V298', 'V18', 'V287', 'V315', 'C7', 'V59', 'V15', 'V52', 'V58', 'V118', 'V320', 'V110',
                    'V78', 'V85', 'V321', 'V312', 'V92', 'V50', 'TransactionDT', 'V127', 'V122', 'V304', 'V81', 'V71', 'V80', 'V74',
                    'V26', 'V116', 'V293', 'V103', 'V310', 'C8', 'V132', 'V51', 'V114', 'V296', 'V68', 'V126'}

        # Columnas no predictivas
        non_predictive_cols = ['TransactionID']

        # Combinamos todas las columnas a eliminar
        all_cols_to_drop = high_cardinality_cols + null_cols + constant_cols + list(corr_cols) + non_predictive_cols

        # Eliminamos las columnas
        X = X.drop(columns=all_cols_to_drop)

        return X

# Funciones de transformación
def log_transform(x):
    return np.log1p(x)

def inverse_log_transform(x):
    x_reflected = x.max() - x    # Reflejamos valores
    return np.log1p(x_reflected)

def sqrt_transform(x):
    return np.sqrt(np.abs(x))


#############################################
#           Variables Globales              #
#############################################

# Columnas categóricas y numéricas
categorical_features = ['ProductCD', 'card4', 'card6', 'M4', 'M6']
numeric_mean_log_cols = ['V66']
numeric_median_log_cols = ['TransactionAmt', 'card3', 'C1', 'C3', 'C5', 'D3', 'V17',
                           'V23', 'V27', 'V37', 'V39', 'V44', 'V46', 'V47', 'V55', 'V56',
                           'V67', 'V77', 'V86', 'V95', 'V98', 'V99', 'V104', 'V108', 'V115',
                           'V117', 'V120', 'V121', 'V129', 'V130', 'V135', 'V281', 'V282', 'V283',
                           'V286', 'V288', 'V290', 'V300', 'V313', 'V314', 'V319']
numeric_median_sqrt_cols = ['card5', 'D1', 'D4', 'D10', 'D15']
numeric_median_inv_cols = ['addr2', 'V14', 'V25', 'V41', 'V107']
numeric_mean_cols = ['card1', 'card2', 'addr1', 'V12', 'V35', 'V53', 'V61', 'V75']
numeric_median_cols = ['V19', 'V29', 'V82']

#############################################
#           Pipelines de Transformación     #
#############################################

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

# Transformador para variables categóricas
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Transformador para variables numéricas con logaritmo (media)
numeric_mean_log_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('log', FunctionTransformer(log_transform))
    # ('scaler', StandardScaler())
])

# Transformador para variables numéricas con logaritmo (mediana)
numeric_median_log_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('log', FunctionTransformer(log_transform))
    # ('scaler', StandardScaler())
])

# Transformador para variables numéricas con raíz cuadrada (mediana)
numeric_median_sqrt_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('sqrt', FunctionTransformer(sqrt_transform))
    # ('scaler', StandardScaler())
])

# Transformador para variables numéricas con transformación inversa logarítmica (mediana)
numeric_median_inv_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('inv', FunctionTransformer(inverse_log_transform))
    # ('scaler', StandardScaler())
])

# Transformador para variables numéricas (media)
numeric_mean_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean'))
    # ('scaler', StandardScaler())
])

# Transformador para variables numéricas (mediana)
numeric_median_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median'))
    # ('scaler', StandardScaler())
])

# Unir los transformadores en un preprocesador
preprocessor = ColumnTransformer(
    transformers=[
        ('num_mean_log', numeric_mean_log_transformer, numeric_mean_log_cols),
        ('num_median_log', numeric_median_log_transformer, numeric_median_log_cols),
        ('num_median_sqrt', numeric_median_sqrt_transformer, numeric_median_sqrt_cols),
        ('num_median_inv', numeric_median_inv_transformer, numeric_median_inv_cols),
        ('num_mean', numeric_mean_transformer, numeric_mean_cols),
        ('num_median', numeric_median_transformer, numeric_median_cols),
        ('cat', categorical_transformer, categorical_features)
    ]
)

# Crear el pipeline de preprocesamiento completo
pipeline_preprocessing = Pipeline(steps=[
    ('drop_columns', DropColumns()),
    ('preprocessor', preprocessor)
])


#############################################
#        Clase ThresholdClassifier          #
#############################################

class ThresholdClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, base_classifier, threshold=0.5):
        self.base_classifier = base_classifier
        self.threshold = threshold

    def fit(self, X, y):
        self.base_classifier.fit(X, y)
        return self

    def predict(self, X):
        probs = self.base_classifier.predict_proba(X)[:, 1]
        return (probs >= self.threshold).astype(int)

    def predict_proba(self, X):
        return self.base_classifier.predict_proba(X)


#############################################
#       Función para construir el pipeline  #
#############################################

def build_final_pipeline(base_classifier, threshold=0.5):
    """
    Construye y devuelve el pipeline final que incluye el preprocesamiento y el clasificador con umbral.
    
    Parameters:
        base_classifier: Un clasificador que tenga el método predict_proba (por ejemplo, XGBoost, RandomForest, etc.).
        threshold (float): Umbral de decisión para convertir probabilidades en clases.
        
    Returns:
        pipeline: Un objeto Pipeline de scikit-learn listo para usarse.
    """
    threshold_classifier = ThresholdClassifier(base_classifier, threshold=threshold)
    
    final_pipeline = Pipeline([
        ('preprocesamiento', pipeline_preprocessing),
        ('modelo', threshold_classifier)
    ])
    
    return final_pipeline

# Fin del archivo custom_pipeline.py
# Este archivo contiene definiciones de transformadores personalizados, funciones de transformación y un pipeline completo
# para preprocesar datos y entrenar un clasificador con umbral.