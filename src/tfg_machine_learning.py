import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import shap
from matplotlib.patches import Patch
from pathlib import Path

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, recall_score, confusion_matrix
from sklearn.metrics import roc_auc_score, precision_score, f1_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = PROJECT_ROOT / "data" / "Heart_disease_cleveland_new.csv"
OUTPUT_DIR = PROJECT_ROOT / "images"
OUTPUT_DIR.mkdir(exist_ok=True)
RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)


# ============================================================
# 1. CARGA Y EXPLORACIÓN DEL DATASET
# ============================================================

dataset = pd.read_csv(DATA_FILE)

print(dataset.describe())
print(dataset.groupby('target').mean())

dataset = dataset.sample(frac=1, random_state=42).reset_index(drop=True)

plt.figure(figsize=(12, 10))
sns.heatmap(dataset.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Graph")
plt.savefig(OUTPUT_DIR / "correlation_matrix.png", dpi=300, bbox_inches="tight")
plt.show()


# ============================================================
# 2. ELIMINAMOS fbs Y chol
# Correlación muy baja con el target (r=0.03 y r=0.09)
# ============================================================

data = dataset.drop(columns=["fbs", "chol"])
data = data.reset_index(drop=True)

X = data.drop(columns=["target"])
y = data["target"]


# ============================================================
# 3. VALIDACIÓN CRUZADA (cv=5)
# ============================================================

print("Naive Bayes CV:",         round(cross_val_score(GaussianNB(), X, y, cv=5).mean(), 4))
print("Random Forest CV:",       round(cross_val_score(RandomForestClassifier(n_estimators=300, random_state=42), X, y, cv=5).mean(), 4))
print("Gradient Boosting CV:",   round(cross_val_score(GradientBoostingClassifier(n_estimators=100, random_state=42), X, y, cv=5).mean(), 4))
print("Regresión Logística CV:", round(cross_val_score(make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000)), X, y, cv=5).mean(), 4))


# ============================================================
# 4. COMPARATIVA CON Y SIN fbs/chol
# ============================================================

X_full = dataset.drop(columns=["target"])
y_full = dataset["target"]

print("\nNaive Bayes con fbs/chol:",      round(cross_val_score(GaussianNB(), X_full, y_full, cv=5).mean(), 4))
print("Naive Bayes sin fbs/chol:",       round(cross_val_score(GaussianNB(), X, y, cv=5).mean(), 4))

print("Random Forest con fbs/chol:",     round(cross_val_score(RandomForestClassifier(n_estimators=300, random_state=42), X_full, y_full, cv=5).mean(), 4))
print("Random Forest sin fbs/chol:",     round(cross_val_score(RandomForestClassifier(n_estimators=300, random_state=42), X, y, cv=5).mean(), 4))

print("Gradient Boosting con fbs/chol:", round(cross_val_score(GradientBoostingClassifier(n_estimators=100, random_state=42), X_full, y_full, cv=5).mean(), 4))
print("Gradient Boosting sin fbs/chol:", round(cross_val_score(GradientBoostingClassifier(n_estimators=100, random_state=42), X, y, cv=5).mean(), 4))

print("Reg. Logística con fbs/chol:",    round(cross_val_score(make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000)), X_full, y_full, cv=5).mean(), 4))
print("Reg. Logística sin fbs/chol:",    round(cross_val_score(make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000)), X, y, cv=5).mean(), 4))


# ============================================================
# 5. DIVISIÓN TRAIN / TEST (80% / 20%)
# ============================================================

corte = int(len(data) * 0.8)

X_train = data[:corte].drop(columns=["target"]).reset_index(drop=True)
y_train = data[:corte]["target"].reset_index(drop=True)

X_test = data[corte:].drop(columns=["target"]).reset_index(drop=True)
y_test = data[corte:]["target"].reset_index(drop=True)


# ============================================================
# 6. NAIVE BAYES MANUAL
# ============================================================

class NaiveBayes:

    def entrenar(self, X, y):
        total = len(y)
        self.prior_probs = {
            0: (y == 0).sum() / total,
            1: (y == 1).sum() / total
        }
        self.conditional_probs = {}
        for clase in [0, 1]:
            self.conditional_probs[clase] = {}
            X_clase = X[y == clase]
            for columna in X.columns:
                self.conditional_probs[clase][columna] = {
                    "mean": X_clase[columna].mean(),
                    "std":  X_clase[columna].std()
                }

    def probabilidad_gaussiana(self, x, media, desv):
        return (1 / (np.sqrt(2 * np.pi) * desv)) * np.exp(-0.5 * ((x - media) / desv) ** 2)

    def predict(self, X):
        predicciones = []
        for _, fila in X.iterrows():
            prob_0 = self.prior_probs[0]
            prob_1 = self.prior_probs[1]
            for columna in X.columns:
                media_0 = self.conditional_probs[0][columna]["mean"]
                desv_0  = self.conditional_probs[0][columna]["std"]
                prob_0  = prob_0 * self.probabilidad_gaussiana(fila[columna], media_0, desv_0)
                media_1 = self.conditional_probs[1][columna]["mean"]
                desv_1  = self.conditional_probs[1][columna]["std"]
                prob_1  = prob_1 * self.probabilidad_gaussiana(fila[columna], media_1, desv_1)
            if prob_0 > prob_1:
                predicciones.append(0)
            else:
                predicciones.append(1)
        return predicciones

modelo_manual = NaiveBayes()
modelo_manual.entrenar(X_train, y_train)
y_pred_manual = modelo_manual.predict(X_test)
print("\nNaive Bayes manual:", round(accuracy_score(y_test, y_pred_manual) * 100, 2), "%")


# ============================================================
# 7. NAIVE BAYES — SKLEARN
# ============================================================

nb = GaussianNB()
nb.fit(X_train, y_train)
y_pred_nb  = nb.predict(X_test)
y_proba_nb = nb.predict_proba(X_test)[:, 1]

print("\n--- Naive Bayes ---")
print("Accuracy:",  round(accuracy_score(y_test, y_pred_nb), 4))
print("Recall:",    round(recall_score(y_test, y_pred_nb), 4))
print("Precision:", round(precision_score(y_test, y_pred_nb), 4))
print("F1:",        round(f1_score(y_test, y_pred_nb), 4))
print("ROC-AUC:",   round(roc_auc_score(y_test, y_proba_nb), 4))

plt.figure(figsize=(6, 5))
sns.heatmap(confusion_matrix(y_test, y_pred_nb), annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicción")
plt.ylabel("Real")
plt.title("Matriz de confusión - Naive Bayes")
plt.savefig(OUTPUT_DIR / "confusion_nb.png", dpi=300, bbox_inches="tight")
plt.close()


# ============================================================
# 8. RANDOM FOREST
# ============================================================

rf = RandomForestClassifier(n_estimators=300, random_state=42)
rf.fit(X_train, y_train)
rf_pred  = list(rf.predict(X_test))
rf_proba = rf.predict_proba(X_test)[:, 1]

print("\n--- Random Forest ---")
print("Accuracy:",  round(accuracy_score(y_test, rf_pred), 4))
print("Recall:",    round(recall_score(y_test, rf_pred), 4))
print("Precision:", round(precision_score(y_test, rf_pred), 4))
print("F1:",        round(f1_score(y_test, rf_pred), 4))
print("ROC-AUC:",   round(roc_auc_score(y_test, rf_proba), 4))

plt.figure(figsize=(6, 5))
sns.heatmap(confusion_matrix(y_test, rf_pred), annot=True, fmt="d", cmap="Greens")
plt.xlabel("Predicción")
plt.ylabel("Real")
plt.title("Matriz de confusión - Random Forest")
plt.savefig(OUTPUT_DIR / "confusion_rf.png", dpi=300, bbox_inches="tight")
plt.close()


# ============================================================
# 9. GRADIENT BOOSTING
# ============================================================

gb = GradientBoostingClassifier(n_estimators=100, random_state=42)
gb.fit(X_train, y_train)
gb_pred  = list(gb.predict(X_test))
gb_proba = gb.predict_proba(X_test)[:, 1]

print("\n--- Gradient Boosting ---")
print("Accuracy:",  round(accuracy_score(y_test, gb_pred), 4))
print("Recall:",    round(recall_score(y_test, gb_pred), 4))
print("Precision:", round(precision_score(y_test, gb_pred), 4))
print("F1:",        round(f1_score(y_test, gb_pred), 4))
print("ROC-AUC:",   round(roc_auc_score(y_test, gb_proba), 4))

plt.figure(figsize=(6, 5))
sns.heatmap(confusion_matrix(y_test, gb_pred), annot=True, fmt="d", cmap="Oranges")
plt.xlabel("Predicción")
plt.ylabel("Real")
plt.title("Matriz de confusión - Gradient Boosting")
plt.savefig(OUTPUT_DIR / "confusion_gb.png", dpi=300, bbox_inches="tight")
plt.close()


# ============================================================
# 10. REGRESIÓN LOGÍSTICA
# ============================================================

logreg = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000))
logreg.fit(X_train, y_train)
log_pred  = list(logreg.predict(X_test))
log_proba = logreg.predict_proba(X_test)[:, 1]

print("\n--- Regresión Logística ---")
print("Accuracy:",  round(accuracy_score(y_test, log_pred), 4))
print("Recall:",    round(recall_score(y_test, log_pred), 4))
print("Precision:", round(precision_score(y_test, log_pred), 4))
print("F1:",        round(f1_score(y_test, log_pred), 4))
print("ROC-AUC:",   round(roc_auc_score(y_test, log_proba), 4))

plt.figure(figsize=(6, 5))
sns.heatmap(confusion_matrix(y_test, log_pred), annot=True, fmt="d", cmap="Purples")
plt.xlabel("Predicción")
plt.ylabel("Real")
plt.title("Matriz de confusión - Regresión Logística")
plt.savefig(OUTPUT_DIR / "confusion_lr.png", dpi=300, bbox_inches="tight")
plt.close()


# ============================================================
# 11. COEFICIENTES DE LA REGRESIÓN LOGÍSTICA
# Positivo = esa variable aumenta el riesgo de enfermedad
# Negativo = esa variable lo reduce
# ============================================================

modelo_lr    = logreg.named_steps["logisticregression"]
coeficientes = modelo_lr.coef_[0]

coef_df = pd.DataFrame({
    "Variable":    list(X_train.columns),
    "Coeficiente": list(coeficientes)
})

print("\n--- Coeficientes Regresión Logística ---")
print(coef_df.round(4).to_string(index=False))
coef_df.to_csv(RESULTS_DIR / "coeficientes_logreg.csv", index=False)

colores_coef = []
for v in coef_df["Coeficiente"]:
    if v > 0:
        colores_coef.append("red")
    else:
        colores_coef.append("blue")

plt.figure(figsize=(8, 5))
plt.barh(coef_df["Variable"], coef_df["Coeficiente"], color=colores_coef)
plt.axvline(0, color="black")
plt.xlabel("Coeficiente")
plt.title("Coeficientes de la Regresión Logística")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "coeficientes_logreg.png", dpi=300, bbox_inches="tight")
plt.close()


# ============================================================
# 12. CALCULAMOS LOS VALORES SHAP
# ============================================================

X_train_scaled = pd.DataFrame(
    logreg.named_steps["standardscaler"].transform(X_train),
    columns=X_train.columns
)

X_test_scaled = pd.DataFrame(
    logreg.named_steps["standardscaler"].transform(X_test),
    columns=X_test.columns
)


def obtener_shap_clase_1(shap_values):
    """
    Devuelve los valores SHAP de la clase positiva (target = 1)
    con forma: pacientes x variables.

    Esto evita errores entre distintas versiones de la librería SHAP:
    - Algunas devuelven una lista: [clase 0, clase 1]
    - Otras devuelven un array 3D: pacientes x variables x clases
    - Otras devuelven directamente un array 2D: pacientes x variables
    """

    if isinstance(shap_values, list):
        return np.array(shap_values[1])

    shap_values = np.array(shap_values)

    if shap_values.ndim == 3:
        return shap_values[:, :, 1]

    return shap_values


def comprobar_dimensiones(nombre_modelo, shap_values, X_referencia):
    """
    Comprueba que SHAP tiene el mismo número de filas y columnas que X_test.
    Así, si algo falla, el error aparece explicado antes de hacer los gráficos.
    """

    if shap_values.shape != X_referencia.shape:
        raise ValueError(
            f"Dimensiones incorrectas en {nombre_modelo}: "
            f"SHAP tiene {shap_values.shape}, pero X_test tiene {X_referencia.shape}"
        )


explainer_rf = shap.TreeExplainer(rf)
shap_rf = obtener_shap_clase_1(explainer_rf.shap_values(X_test))

explainer_gb = shap.TreeExplainer(gb)
shap_gb = obtener_shap_clase_1(explainer_gb.shap_values(X_test))

explainer_lr = shap.LinearExplainer(modelo_lr, X_train_scaled)
shap_lr = obtener_shap_clase_1(explainer_lr.shap_values(X_test_scaled))

# Número de pacientes en el test
n = len(X_test)

# Comprobación de seguridad antes de pintar los gráficos
comprobar_dimensiones("Random Forest", shap_rf, X_test)
comprobar_dimensiones("Gradient Boosting", shap_gb, X_test)
comprobar_dimensiones("Regresión Logística", shap_lr, X_test_scaled)

print("Filas shap_rf:", shap_rf.shape)
print("Filas shap_gb:", shap_gb.shape)
print("Filas shap_lr:", shap_lr.shape)
print("Filas X_test:", X_test.shape)

# ============================================================
# 13. BEESWARM COLOREADO POR CLASE PREDICHA — RANDOM FOREST
# Cada punto es un paciente.
# Rojo = el modelo predijo enfermedad. Azul = el modelo predijo sano.
# ============================================================

colores_rf = []
for pred in rf_pred:
    if pred == 1:
        colores_rf.append("#d73027")
    else:
        colores_rf.append("#4575b4")
colores_rf = np.array(colores_rf)

importancia_rf = np.abs(shap_rf).mean(axis=0)
orden_rf = np.argsort(importancia_rf)
variables_rf = list(X_test.columns[orden_rf])

plt.figure(figsize=(8, 6))
for pos in range(len(variables_rf)):
    variable = variables_rf[pos]
    idx = list(X_test.columns).index(variable)
    ruido = np.random.normal(0, 0.08, size=n)
    y_pos = np.ones(n) * pos + ruido
    plt.scatter(shap_rf[:, idx], y_pos, c=colores_rf, alpha=0.7, s=20)

plt.yticks(range(len(variables_rf)), variables_rf)
plt.axvline(0, color="black")
plt.xlabel("Valor SHAP")
plt.title("SHAP - Random Forest (rojo=enfermedad, azul=sano)")
leyenda = [Patch(facecolor="#d73027", label="Predicho: enfermedad"),
           Patch(facecolor="#4575b4", label="Predicho: sano")]
plt.legend(handles=leyenda, loc="lower right")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "shap_rf_beeswarm.png", dpi=300, bbox_inches="tight")
plt.close()


# ============================================================
# 14. BEESWARM COLOREADO POR CLASE PREDICHA — GRADIENT BOOSTING
# ============================================================

colores_gb = []
for pred in gb_pred:
    if pred == 1:
        colores_gb.append("#d73027")
    else:
        colores_gb.append("#4575b4")
colores_gb = np.array(colores_gb)

importancia_gb = np.abs(shap_gb).mean(axis=0)
orden_gb = np.argsort(importancia_gb)
variables_gb = list(X_test.columns[orden_gb])

plt.figure(figsize=(8, 6))
for pos in range(len(variables_gb)):
    variable = variables_gb[pos]
    idx = list(X_test.columns).index(variable)
    ruido = np.random.normal(0, 0.08, size=n)
    y_pos = np.ones(n) * pos + ruido
    plt.scatter(shap_gb[:, idx], y_pos, c=colores_gb, alpha=0.7, s=20)

plt.yticks(range(len(variables_gb)), variables_gb)
plt.axvline(0, color="black")
plt.xlabel("Valor SHAP")
plt.title("SHAP - Gradient Boosting (rojo=enfermedad, azul=sano)")
leyenda = [Patch(facecolor="#d73027", label="Predicho: enfermedad"),
           Patch(facecolor="#4575b4", label="Predicho: sano")]
plt.legend(handles=leyenda, loc="lower right")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "shap_gb_beeswarm.png", dpi=300, bbox_inches="tight")
plt.close()


# ============================================================
# 15. BEESWARM COLOREADO POR CLASE PREDICHA — REGRESIÓN LOGÍSTICA
# ============================================================

colores_lr = []
for pred in log_pred:
    if pred == 1:
        colores_lr.append("#d73027")
    else:
        colores_lr.append("#4575b4")
colores_lr = np.array(colores_lr)

importancia_lr_orden = np.abs(shap_lr).mean(axis=0)
orden_lr = np.argsort(importancia_lr_orden)
variables_lr = list(X_test_scaled.columns[orden_lr])

plt.figure(figsize=(8, 6))
for pos in range(len(variables_lr)):
    variable = variables_lr[pos]
    idx = list(X_test_scaled.columns).index(variable)
    ruido = np.random.normal(0, 0.08, size=n)
    y_pos = np.ones(n) * pos + ruido
    plt.scatter(shap_lr[:, idx], y_pos, c=colores_lr, alpha=0.7, s=20)

plt.yticks(range(len(variables_lr)), variables_lr)
plt.axvline(0, color="black")
plt.xlabel("Valor SHAP")
plt.title("SHAP - Regresión Logística (rojo=enfermedad, azul=sano)")
leyenda = [Patch(facecolor="#d73027", label="Predicho: enfermedad"),
           Patch(facecolor="#4575b4", label="Predicho: sano")]
plt.legend(handles=leyenda, loc="lower right")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "shap_lr_beeswarm.png", dpi=300, bbox_inches="tight")
plt.close()


# ============================================================
# 16. FIGURA COMPACTA: LOS 3 BEESWARMS EN UNA SOLA FILA
# ============================================================

fig, axes = plt.subplots(1, 3, figsize=(20, 7))

for pos in range(len(variables_rf)):
    variable = variables_rf[pos]
    idx = list(X_test.columns).index(variable)
    ruido = np.random.normal(0, 0.08, size=n)
    axes[0].scatter(shap_rf[:, idx], np.ones(n) * pos + ruido, c=colores_rf, alpha=0.7, s=20)
axes[0].set_yticks(range(len(variables_rf)))
axes[0].set_yticklabels(variables_rf)
axes[0].axvline(0, color="black")
axes[0].set_title("Random Forest")
axes[0].set_xlabel("Valor SHAP")

for pos in range(len(variables_gb)):
    variable = variables_gb[pos]
    idx = list(X_test.columns).index(variable)
    ruido = np.random.normal(0, 0.08, size=n)
    axes[1].scatter(shap_gb[:, idx], np.ones(n) * pos + ruido, c=colores_gb, alpha=0.7, s=20)
axes[1].set_yticks(range(len(variables_gb)))
axes[1].set_yticklabels(variables_gb)
axes[1].axvline(0, color="black")
axes[1].set_title("Gradient Boosting")
axes[1].set_xlabel("Valor SHAP")

for pos in range(len(variables_lr)):
    variable = variables_lr[pos]
    idx = list(X_test_scaled.columns).index(variable)
    ruido = np.random.normal(0, 0.08, size=n)
    axes[2].scatter(shap_lr[:, idx], np.ones(n) * pos + ruido, c=colores_lr, alpha=0.7, s=20)
axes[2].set_yticks(range(len(variables_lr)))
axes[2].set_yticklabels(variables_lr)
axes[2].axvline(0, color="black")
axes[2].set_title("Regresión Logística")
axes[2].set_xlabel("Valor SHAP")

leyenda = [Patch(facecolor="#d73027", label="Predicho: enfermedad"),
           Patch(facecolor="#4575b4", label="Predicho: sano")]
fig.legend(handles=leyenda, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.02))
fig.suptitle("SHAP — Los 3 modelos (coloreado por clase predicha)")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "shap_beeswarms_compacto.png", dpi=300, bbox_inches="tight")
plt.close()


# ============================================================
# 17. TABLA DE IMPORTANCIA SHAP
# ============================================================

importancia_rf = np.abs(shap_rf).mean(axis=0)
importancia_gb = np.abs(shap_gb).mean(axis=0)
importancia_lr = np.abs(shap_lr).mean(axis=0)

tabla_shap = pd.DataFrame({
    "Variable":       list(X_test.columns),
    "Random Forest":  list(importancia_rf),
    "Grad. Boosting": list(importancia_gb),
    "Logistic Reg.":  list(importancia_lr)
})

print("\n--- Importancia media SHAP ---")
print(tabla_shap.round(4).to_string(index=False))
tabla_shap.to_csv(RESULTS_DIR / "shap_importancia_tabla.csv", index=False)


# ============================================================
# 18. COMPARATIVA SHAP: DISEASE VS NO DISEASE
# Separamos los pacientes por lo que predijo el modelo
# y calculamos la media SHAP de cada grupo.
# ============================================================

# Random Forest
idx_disease_rf    = [i for i in range(len(rf_pred)) if rf_pred[i] == 1]
idx_no_disease_rf = [i for i in range(len(rf_pred)) if rf_pred[i] == 0]

tabla_rf = pd.DataFrame({
    "Variable":                list(X_test.columns),
    "SHAP medio (disease)":    list(shap_rf[idx_disease_rf].mean(axis=0)),
    "SHAP medio (no disease)": list(shap_rf[idx_no_disease_rf].mean(axis=0))
})
tabla_rf["Diferencia"] = tabla_rf["SHAP medio (disease)"] - tabla_rf["SHAP medio (no disease)"]
print("\n--- SHAP disease vs no disease: Random Forest ---")
print(tabla_rf.round(4).to_string(index=False))
tabla_rf.to_csv(RESULTS_DIR / "shap_grupos_rf.csv", index=False)

# Gradient Boosting
idx_disease_gb    = [i for i in range(len(gb_pred)) if gb_pred[i] == 1]
idx_no_disease_gb = [i for i in range(len(gb_pred)) if gb_pred[i] == 0]

tabla_gb = pd.DataFrame({
    "Variable":                list(X_test.columns),
    "SHAP medio (disease)":    list(shap_gb[idx_disease_gb].mean(axis=0)),
    "SHAP medio (no disease)": list(shap_gb[idx_no_disease_gb].mean(axis=0))
})
tabla_gb["Diferencia"] = tabla_gb["SHAP medio (disease)"] - tabla_gb["SHAP medio (no disease)"]
print("\n--- SHAP disease vs no disease: Gradient Boosting ---")
print(tabla_gb.round(4).to_string(index=False))
tabla_gb.to_csv(RESULTS_DIR / "shap_grupos_gb.csv", index=False)

# Regresión Logística
idx_disease_lr    = [i for i in range(len(log_pred)) if log_pred[i] == 1]
idx_no_disease_lr = [i for i in range(len(log_pred)) if log_pred[i] == 0]

tabla_lr = pd.DataFrame({
    "Variable":                list(X_test_scaled.columns),
    "SHAP medio (disease)":    list(shap_lr[idx_disease_lr].mean(axis=0)),
    "SHAP medio (no disease)": list(shap_lr[idx_no_disease_lr].mean(axis=0))
})
tabla_lr["Diferencia"] = tabla_lr["SHAP medio (disease)"] - tabla_lr["SHAP medio (no disease)"]
print("\n--- SHAP disease vs no disease: Regresión Logística ---")
print(tabla_lr.round(4).to_string(index=False))
tabla_lr.to_csv(RESULTS_DIR / "shap_grupos_lr.csv", index=False)


# ============================================================
# 19. COMPARATIVA SHAP vs COEFICIENTES CON SIGNO
# Comprobamos si SHAP y coeficiente van en la misma dirección
# ============================================================

shap_lr_medio = shap_lr.mean(axis=0)

print("\n--- Comparativa SHAP (con signo) vs Coeficientes ---")

for i in range(len(X_test_scaled.columns)):
    variable    = X_test_scaled.columns[i]
    shap_valor  = shap_lr_medio[i]
    coeficiente = coef_df[coef_df["Variable"] == variable]["Coeficiente"].values[0]

    if coeficiente > 0:
        signo_coef = "positivo"
    else:
        signo_coef = "negativo"

    if shap_valor > 0:
        signo_shap = "positivo"
    else:
        signo_shap = "negativo"

    if signo_coef == signo_shap:
        coinciden = "SI"
    else:
        coinciden = "NO"

    print(variable, "— Coef:", round(coeficiente, 4),
          "| SHAP medio:", round(shap_valor, 4),
          "| Misma dirección:", coinciden)
