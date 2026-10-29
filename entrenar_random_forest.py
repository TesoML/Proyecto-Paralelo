import os
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score

def entrenar_rf():
    ruta_base = r"C:\Users\eliye\OneDrive\Documentos\SextoSemestre\ComputoParalelo\Proyecto-Paralelo"
    ruta_csv = os.path.join(ruta_base, "resultados_fusion_hibrido.csv")

    if not os.path.exists(ruta_csv):
        print(f"No se encontró el archivo fusionado en: {ruta_csv}")
        return


    df = pd.read_csv(ruta_csv).dropna()

    features = [col for col in df.columns if col not in ['slice_file_name', 'fold', 'classID', 'class']]
    X = df[features]
    y = df['class']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    t_inicio = time.perf_counter()
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf_model.fit(X_train_scaled, y_train)
    t_fin = time.perf_counter()

    tiempo_entrenamiento = t_fin - t_inicio
    y_pred = rf_model.predict(X_test_scaled)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

    print(f"  • Tiempo de Entrenamiento: {tiempo_entrenamiento:.4f} s")
    print(f"  • Exactitud (Accuracy): {acc * 100:.2f}% | F1-Score: {f1:.4f}")

    df_res = pd.DataFrame([{
        'Modelo': 'Random Forest', 
        'Tiempo_Entrenamiento(s)': tiempo_entrenamiento, 
        'Exactitud': acc, 
        'Precision': prec, 
        'Recall': rec, 
        'F1_Score': f1
    }])
    df_res.to_csv(os.path.join(ruta_base, "metrica_random_forest.csv"), index=False)
    print("¡Métricas de Random Forest guardadas con éxito!")

if __name__ == '__main__':
    entrenar_rf()
