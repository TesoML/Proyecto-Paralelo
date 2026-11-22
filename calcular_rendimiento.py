
def calcular_metricas_paralelas():
    # Tiempos secuenciales (1 proceso)
    t1_stft = 1114.7404
    t1_mfcc = 1141.5130

    # Tiempos paralelos (con 2 procesos / hilos)
    tp_stft = 422.6020
    tp_mfcc = 392.4502

    # Número de procesos utilizados (N)
    N = 2

    # STFT Metricas
    speedup_stft = t1_stft / tp_stft
    eficiencia_stft = (speedup_stft / N) * 100

    # MFCC Metricas
    speedup_mfcc = t1_mfcc / tp_mfcc
    eficiencia_mfcc = (speedup_mfcc / N) * 100

    print("\n[1. Extracción STFT]")
    print(f"  • Tiempo Secuencial (T1): {t1_stft:.4f} s")
    print(f"  • Tiempo Paralelo (Tp):   {tp_stft:.4f} s")
    print(f"  • Speedup (S):            {speedup_stft:.4f}x")
    print(f"  • Eficiencia:             {eficiencia_stft:.2f}%")

    print("\n[2. Extracción MFCC]")
    print(f"  • Tiempo Secuencial (T1): {t1_mfcc:.4f} s")
    print(f"  • Tiempo Paralelo (Tp):   {tp_mfcc:.4f} s")
    print(f"  • Speedup (S):            {speedup_mfcc:.4f}x")
    print(f"  • Eficiencia:             {eficiencia_mfcc:.2f}%")

if __name__ == '__main__':
    calcular_metricas_paralelas()
