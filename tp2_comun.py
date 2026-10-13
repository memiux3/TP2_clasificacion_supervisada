"""Definiciones compartidas entre los notebooks del TP2 (Bank Marketing).

Todo lo que está acá son reglas FIJAS decididas en el Paso 1 (EDA sobre train):
no aprenden nada de los datos, así que pueden aplicarse a train y test sin leakage.
Las transformaciones que SÍ aprenden (escalado, One-Hot) se crean con
`crear_preprocesador()` y se ajustan siempre dentro de un Pipeline.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

RANDOM_STATE = 42
DATA_DIR = Path("data")
FIG = Path("figuras")
RESULTADOS = Path("resultados")

# ---------------------------------------------------------------- datos
def cargar_split(nombre):
    """Lee la partición guardada en el Paso 1 ('train' o 'test') -> (X, y)."""
    df = pd.read_csv(DATA_DIR / f"{nombre}.csv")
    return df.drop(columns="y"), df["y"]


# ---------------------------------------------------------------- features
CORTES_EDAD = [0, 24, 60, 200]
ETIQ_EDAD = ["<=24", "25-60", ">60"]


def agregar_features(X):
    """Variables nuevas del EDA (reglas fila a fila, sin leakage)."""
    X = X.copy()
    X["contactado_antes"] = (X["pdays"] != 999).astype(int)
    X["edad_grupo"] = pd.cut(X["age"], bins=CORTES_EDAD, labels=ETIQ_EDAD).astype(str)
    return X


NUM_B = ["age", "campaign", "previous", "cons.price.idx", "cons.conf.idx", "nr.employed", "contactado_antes"]
CAT_B = ["job", "marital", "education", "default", "contact", "month", "poutcome"]
CONTEXTO = ["month", "nr.employed", "cons.price.idx", "cons.conf.idx"]

SETS = {
    "A · Completo": {
        "num": ["age", "campaign", "pdays", "previous", "emp.var.rate", "cons.price.idx",
                "cons.conf.idx", "euribor3m", "nr.employed"],
        "cat": ["job", "marital", "education", "default", "housing", "loan", "contact",
                "month", "day_of_week", "poutcome"],
    },
    "B · EDA": {"num": NUM_B, "cat": CAT_B},
    "B + edad": {"num": NUM_B, "cat": CAT_B + ["edad_grupo"]},
    "C · Sin contexto": {
        "num": [c for c in NUM_B if c not in CONTEXTO],
        "cat": [c for c in CAT_B if c not in CONTEXTO],
    },
}


def columnas(set_nombre):
    return SETS[set_nombre]["num"] + SETS[set_nombre]["cat"]


def crear_preprocesador(set_nombre):
    """Escalado z-score + One-Hot. Se ajusta SOLO dentro del Pipeline (con train / folds de train)."""
    s = SETS[set_nombre]
    return ColumnTransformer([
        ("num", StandardScaler(), s["num"]),
        ("cat", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=100,
                              sparse_output=False), s["cat"]),
    ])


# ---------------------------------------------------------------- gráficos
C_NO, C_YES, C_GRIS = "#2a78d6", "#eb6834", "#8a8984"
PAL_Y = {"no": C_NO, "yes": C_YES}
# Colores categóricos en orden fijo (uno por set, siempre el mismo color para el mismo set)
COLORES_SET = {"A · Completo": "#2a78d6", "B · EDA": "#eb6834", "B + edad": "#1baf7a", "C · Sin contexto": "#eda100"}
# Un color fijo por modelo
COLORES_MODELO = {"Naive Bayes": "#2a78d6", "SVM": "#eb6834", "KNN": "#1baf7a", "Random Forest": "#4a3aa7"}


def estilo_graficos():
    plt.rcParams.update({
        "figure.dpi": 110, "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": "#e6e5e0", "grid.linewidth": 0.6, "axes.axisbelow": True,
        "axes.edgecolor": "#8a8984", "axes.labelcolor": "#2b2b2a", "xtick.color": "#52514e",
        "ytick.color": "#52514e", "axes.titleweight": "bold", "font.size": 10,
    })


def guardar(nombre):
    FIG.mkdir(exist_ok=True)
    plt.tight_layout()
    plt.savefig(FIG / f"{nombre}.png", dpi=150, bbox_inches="tight")
    plt.show()
