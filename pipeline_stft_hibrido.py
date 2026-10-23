import os
import time
import numpy as np
import pandas as pd
import librosa
from mpi4py import MPI
from mpi4py.MPI import COMM_WORLD
from multiprocessing import Pool
import socket

def extraer_stft_individual(row_data):
    ruta_base, fold, nombre_archivo, class_id, clase_nombre = row_data
    ruta_audio = os.path.join(ruta_base, "UrbanSound8K", "audio", f"fold{fold}", nombre_archivo)

    if not os.path.exists(ruta_audio):
        return None

    try:
        y, sr = librosa.load(ruta_audio, sr=None)
        stft_matrix = librosa.stft(y, n_fft=2048, hop_length=512)
        stft_magnitude = np.abs(stft_matrix)

        return {
            'slice_file_name': nombre_archivo,
            'fold': fold,
            'classID': class_id,
            'class': clase_nombre,
            'stft_mean': np.mean(stft_magnitude),
            'stft_std': np.std(stft_magnitude),
            'stft_max': np.max(stft_magnitude),
            'stft_min': np.min(stft_magnitude)
        }
    except Exception as e:
        return None

def main():
    comm = COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    hostname = socket.gethostname()

    comm.Barrier()
    tiempo_inicio = MPI.Wtime()  # Corregido con MPI.Wtime()

    ruta_base = r"C:\Users\eliye\OneDrive\Documentos\SextoSemestre\ComputoParalelo\Proyecto-Paralelo"
    ruta_csv = os.path.join(ruta_base, "UrbanSound8K", "metadata", "UrbanSound8K.csv")

    if rank == 0:
        print(f"=== INICIANDO EXTRACCIÓN HÍBRIDA: STFT ===")
        print(f"Procesos MPI activos: {size} | Máquina: {hostname}")
        df_meta = pd.read_csv(ruta_csv)

        # Empaquetamos los datos necesarios de cada fila
        registros = list(zip(
            [ruta_base]*len(df_meta), 
            df_meta['fold'], 
            df_meta['slice_file_name'],
            df_meta['classID'],
            df_meta['class']
        ))
    else:
        registros = None

    registros = comm.bcast(registros, root=0)

    # Reparto MPI
    lotes_por_proceso = np.array_split(registros, size)
    mi_lote = lotes_por_proceso[rank].tolist()

    print(f"-> [MPI Rank {rank}] Procesando {len(mi_lote)} archivos (STFT).")

    # Reparto Multiprocessing
    num_hilos_internos = 2
    resultados_locales = []

    with Pool(processes=num_hilos_internos) as pool:
        resultados_locales = pool.map(extraer_stft_individual, mi_lote)

    resultados_locales = [res for res in resultados_locales if res is not None]

    comm.Barrier()
    tiempo_fin = MPI.Wtime()  # Corregido con MPI.Wtime()
    tiempo_local = tiempo_fin - tiempo_inicio

    todos_los_tiempos = comm.gather(tiempo_local, root=0)
    todos_los_resultados = comm.gather(resultados_locales, root=0)

    if rank == 0:
        print(f"\n[Resultados de Rendimiento - STFT]:")
        for p_id, t in enumerate(todos_los_tiempos):
            print(f"  • Proceso MPI {p_id}: {t:.4f} segundos")

        tiempo_maximo = max(todos_los_tiempos)
        print(f"\n> Tiempo Total Paralelo (STFT): {tiempo_maximo:.4f} segundos ({tiempo_maximo/60:.2f} minutos)")

        resultados_totales = [item for sublist in todos_los_resultados for item in sublist]
        df_final = pd.DataFrame(resultados_totales)
        ruta_salida = os.path.join(ruta_base, "resultados_stft_hibrido.csv")
        df_final.to_csv(ruta_salida, index=False)
        print(f"¡Guardado exitosamente en: {ruta_salida}!")

if __name__ == '__main__':
    main()
