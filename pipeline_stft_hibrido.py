import os
import socket
import numpy as np
import pandas as pd
import librosa
from mpi4py import MPI
from multiprocessing import Pool

RUTA_DATASET = os.path.expanduser("~/cluster_mpi/UrbanSound8K")
RUTA_SALIDA = os.path.expanduser("~/cluster_mpi/resultados_stft_hibrido.csv")

def extraer_stft_individual(row_data):
    fold, nombre_archivo, class_id, clase_nombre = row_data
    ruta_audio = os.path.join(RUTA_DATASET, "audio", f"fold{fold}", nombre_archivo)

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
            'stft_mean': float(np.mean(stft_magnitude)),
            'stft_std': float(np.std(stft_magnitude)),
            'stft_max': float(np.max(stft_magnitude)),
            'stft_min': float(np.min(stft_magnitude))
        }
    except Exception:
        return None

def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    hostname = socket.gethostname()

    comm.Barrier()
    tiempo_inicio = MPI.Wtime()

    ruta_csv = os.path.join(RUTA_DATASET, "metadata", "UrbanSound8K.csv")

    if rank == 0:
        print("=== INICIANDO EXTRACCIÓN HÍBRIDA: STFT ===", flush=True)
        print(f"Procesos MPI activos: {size} | Nodo Maestro: {hostname}", flush=True)
        df_meta = pd.read_csv(ruta_csv)

        registros = list(zip(
            df_meta['fold'].tolist(),
            df_meta['slice_file_name'].tolist(),
            df_meta['classID'].tolist(),
            df_meta['class'].tolist()
        ))
    else:
        registros = None

    registros = comm.bcast(registros, root=0)

    # Reparto balanceado entre nodos MPI
    mi_lote = registros[rank::size]
    print(f"-> [Rank {rank} @ {hostname}] Procesando {len(mi_lote)} audios (STFT).", flush=True)

    # 2 trabajadores locales por proceso MPI
    num_hilos_internos = 2
    with Pool(processes=num_hilos_internos) as pool:
        resultados_locales = pool.map(extraer_stft_individual, mi_lote)

    resultados_locales = [res for res in resultados_locales if res is not None]

    comm.Barrier()
    tiempo_fin = MPI.Wtime()
    tiempo_local = tiempo_fin - tiempo_inicio

    todos_los_tiempos = comm.gather((rank, hostname, tiempo_local), root=0)
    todos_los_resultados = comm.gather(resultados_locales, root=0)

    if rank == 0:
        print("\n[Resultados de Rendimiento - STFT]:")
        for p_id, nodo, t in todos_los_tiempos:
            print(f"  • Rank {p_id} ({nodo}): {t:.4f} segundos")

        tiempo_maximo = max(t for _, _, t in todos_los_tiempos)
        print(f"\n> Tiempo Total Paralelo (STFT): {tiempo_maximo:.4f} segundos ({tiempo_maximo/60:.2f} minutos)")

        resultados_totales = [item for sublist in todos_los_resultados for item in sublist]
        df_final = pd.DataFrame(resultados_totales)
        df_final.to_csv(RUTA_SALIDA, index=False)
        print(f"¡Guardado exitosamente ({len(df_final)} filas) en: {RUTA_SALIDA}!")

if __name__ == '__main__':
    main()
