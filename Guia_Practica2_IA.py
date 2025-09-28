# --------------------------
# Algoritmo de Dijkstra
# --------------------------
import heapq

def dijkstra(grafo, inicio):
    # Distancias iniciales: infinito para todos menos el nodo de inicio
    distancias = {nodo: float('inf') for nodo in grafo}
    distancias[inicio] = 0

    # Cola de prioridad para elegir el nodo más cercano
    cola_prioridad = [(0, inicio)]

    while cola_prioridad:
        distancia_actual, nodo_actual = heapq.heappop(cola_prioridad)

        # Si encontramos una distancia mayor, no es óptima y la ignoramos
        if distancia_actual > distancias[nodo_actual]:
            continue

        # Explorar vecinos
        for vecino, peso in grafo[nodo_actual].items():
            distancia = distancia_actual + peso

            # Si encontramos una ruta más corta, actualizamos
            if distancia < distancias[vecino]:
                distancias[vecino] = distancia
                heapq.heappush(cola_prioridad, (distancia, vecino))

    return distancias


grafo = {
    'A': {'B': 2, 'C': 5},
    'B': {'A': 2, 'C': 6, 'D': 1},
    'C': {'A': 5, 'B': 6, 'D': 2, 'E': 5},
    'D': {'B': 1, 'C': 2, 'E': 1},
    'E': {'C': 5, 'D': 1}
}

resultado = dijkstra(grafo, 'A')
print("Distancias mínimas desde A:")
for nodo, distancia in resultado.items():
    print(f"A → {nodo}: {distancia}")


# --------------------------
# Algoritmo de Floyd-Warshall
# --------------------------
def floyd_warshall(grafo):
    # Número de nodos
    nodos = list(grafo.keys())
    n = len(nodos)

    # Inicializar matriz de distancias
    dist = [[float('inf')] * n for _ in range(n)]

    # Distancia a sí mismo = 0
    for i in range(n):
        dist[i][i] = 0

    # Llenar con valores del grafo
    for u in grafo:
        for v, peso in grafo[u].items():
            dist[nodos.index(u)][nodos.index(v)] = peso

    # Algoritmo principal
    for k in range(n):
        for i in range(n):
            for j in range(n):
                if dist[i][k] + dist[k][j] < dist[i][j]:
                    dist[i][j] = dist[i][k] + dist[k][j]

    # Mostrar resultado
    print("\nMatriz de distancias mínimas (Floyd-Warshall):")
    for fila in dist:
        print(fila)


# Grafo de ejemplo para Floyd-Warshall
grafo_fw = {
    'A': {'B': 3, 'C': 8},
    'B': {'C': 2, 'D': 5},
    'C': {'D': 1},
    'D': {'A': 2}
}
floyd_warshall(grafo_fw)


# --------------------------
# Búsqueda en Anchura (BFS)
# --------------------------
from collections import deque

def bfs(grafo, inicio, meta):
    visitados = set()
    cola = deque([[inicio]])

    while cola:
        camino = cola.popleft()
        nodo = camino[-1]

        if nodo == meta:
            return camino

        if nodo not in visitados:
            for vecino in grafo[nodo]:
                nuevo_camino = list(camino)
                nuevo_camino.append(vecino)
                cola.append(nuevo_camino)

            visitados.add(nodo)

    return None


# Grafo de ejemplo para BFS
grafo_bfs = {
    'A': ['B', 'C'],
    'B': ['A', 'D', 'E'],
    'C': ['A', 'F'],
    'D': ['B'],
    'E': ['B', 'F'],
    'F': ['C', 'E']
}

print("\nCamino BFS de A a F:", bfs(grafo_bfs, 'A', 'F'))
