
import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.multioutput import MultiOutputClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.preprocessing import MultiLabelBinarizer

from sklearn.metrics import classification_report

# ==========================
# CARGAR DATASET
# ==========================

df = pd.read_csv("dataset_with_diagnostico.csv")

# ==========================
# TEXTO
# ==========================

X_texto = df["entrada"]

# ==========================
# TF-IDF
# ==========================

vectorizador = TfidfVectorizer(
    max_features=1000
)

X = vectorizador.fit_transform(X_texto)

# ==========================
# ETIQUETAS MULTI-LABEL
# ==========================

y = df["diagnostivo"].apply(
    lambda x: x.split(",")
)

# ==========================
# CONVERTIR A BINARIO
# ==========================

mlb = MultiLabelBinarizer()

Y = mlb.fit_transform(y)

# ==========================
# DIVISIÓN
# ==========================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    Y,
    test_size=0.2,
    random_state=42
)

# ==========================
# MODELO
# ==========================

modelo = MultiOutputClassifier(
    RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )
)

# ==========================
# ENTRENAMIENTO
# ==========================

modelo.fit(X_train, y_train)

# ==========================
# PREDICCIONES
# ==========================

predicciones = modelo.predict(X_test)

import matplotlib.pyplot as plt

from sklearn.metrics import (
confusion_matrix,
ConfusionMatrixDisplay,
accuracy_score,
f1_score,
classification_report
)


accuracy = accuracy_score(
y_test,
predicciones
)

print("\n=========================")
print("ACCURACY")
print("=========================")

print(
f"Accuracy: {accuracy:.4f}"
)


f1_micro = f1_score(
y_test,
predicciones,
average='micro'
)

f1_macro = f1_score(
y_test,
predicciones,
average='macro'
)

print("\n=========================")
print("F1 SCORE")
print("=========================")

print(
f"F1 Micro: {f1_micro:.4f}"
)

print(
f"F1 Macro: {f1_macro:.4f}"
)



print("\n=========================")
print("MATRICES DE CONFUSION")
print("=========================")

for i, etiqueta in enumerate(mlb.classes_):

    # REAL
    y_real = y_test[:, i]

    # PREDICCION
    y_pred = predicciones[:, i]

    # MATRIZ
    cm = confusion_matrix(
        y_real,
        y_pred
    )

    print(f"\nEtiqueta: {etiqueta}")
    print(cm)

    # VISUALIZACION
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=[
            "No",
            "Si"
        ]
    )

    disp.plot(
        cmap=plt.cm.Blues
    )

    plt.title(
        f"Matriz de Confusión - {etiqueta}"
    )

    plt.show()


print("\n=========================")
print("CLASSIFICATION REPORT")
print("=========================")

print(
classification_report(
y_test,
predicciones,
target_names=mlb.classes_
)
)




