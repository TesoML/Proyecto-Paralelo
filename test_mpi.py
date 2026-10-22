from mpi4py.MPI import COMM_WORLD
import socket

# Inicializamos el comunicador global
comm = COMM_WORLD

# Obtenemos el Rank y el tamaño total de procesos
rank = comm.Get_rank()
size = comm.Get_size()

# Obtenemos el nombre de la máquina
hostname = socket.gethostname()

print(f"Hola desde el proceso {rank} de {size} corriendo en la maquina: {hostname}")

if rank == 0:
    print(f"\n[Maestro - Rank 0] ¡Hola! Coordinando la ejecucion de {size} procesos.")
    data = {"archivo": "audio_01.wav", "etiqueta": "siren"}
    if size > 1:
        comm.send(data, dest=1, tag=11)
        print("[Maestro - Rank 0] Datos de audio enviados al proceso 1.")
else:
    if rank == 1:
        received_data = comm.recv(source=0, tag=11)
        print(f"   [Proceso 1] Mensaje recibido del maestro: {received_data}")