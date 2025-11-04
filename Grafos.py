import time
import heapq
import random
from collections import deque
import math
import pandas as pd

# ==========================
# CLASSE GRAFO
# ==========================
class Grafo:
    def __init__(self, arquivo_arestas, representacao='lista', limit_vertices=None):
        self.arestas = []
        self.representacao = representacao
        self.limit_vertices = limit_vertices
        self.lerArquivo(arquivo_arestas)
        self.escolherRepresentacao(representacao)

    def lerArquivo(self, arquivo_arestas):
        with open(arquivo_arestas, "r") as file:
            linhas = file.readlines()

        # Lê número de vértices
        for line in linhas:
            if line.strip():
                self.quantidade_vertices = int(line.strip())
                break

        # Se o grafo for muito grande, podemos limitar (opcional)
        if self.limit_vertices:
            self.quantidade_vertices_effective = min(self.quantidade_vertices, self.limit_vertices)
        else:
            self.quantidade_vertices_effective = self.quantidade_vertices

        self.graus = [0] * self.quantidade_vertices_effective
        soma_graus = 0

        # Lê as arestas
        for line in linhas[1:]:
            if not line.strip():
                continue
            parts = line.split()
            if len(parts) < 2:
                continue

            u = int(parts[0])
            v = int(parts[1])
            if u > self.quantidade_vertices_effective or v > self.quantidade_vertices_effective:
                continue

            if len(parts) == 3:
                w = float(parts[2])
                if w < 0:
                    raise Exception("Pesos negativos não suportados.")
                self.arestas.append((u, v, w))
            else:
                self.arestas.append((u, v, 1.0))

            self.graus[u - 1] += 1
            self.graus[v - 1] += 1
            soma_graus += 2

        self.quantidade_arestas = len(self.arestas)
        self.grau_medio = soma_graus / self.quantidade_vertices_effective

    def criarListaAdjacencia(self):
        lista = [deque() for _ in range(self.quantidade_vertices_effective)]
        for u, v, w in self.arestas:
            lista[u - 1].append((v, w))
            lista[v - 1].append((u, w))
        return lista

    def criarMatrizAdjacencia(self):
        n = self.quantidade_vertices_effective
        matriz = [[0] * n for _ in range(n)]
        for u, v, w in self.arestas:
            matriz[u - 1][v - 1] = w
            matriz[v - 1][u - 1] = w
        return matriz

    def escolherRepresentacao(self, representacao):
        if representacao == 'matriz':
            self.estruturaGrafo = self.criarMatrizAdjacencia()
        elif representacao == 'lista':
            self.estruturaGrafo = self.criarListaAdjacencia()
        else:
            raise Exception("Representação inválida")

    # Dijkstra com Heap (eficiente)
    def dijkstraHeap(self, vertice_inicial):
        n = self.quantidade_vertices_effective
        dist = [float('inf')] * n
        prev = [None] * n
        dist[vertice_inicial - 1] = 0
        heap = [(0, vertice_inicial)]

        while heap:
            d_u, u = heapq.heappop(heap)
            if d_u > dist[u - 1]:
                continue
            for v, w in self.estruturaGrafo[u - 1]:
                if dist[v - 1] > d_u + w:
                    dist[v - 1] = d_u + w
                    prev[v - 1] = u
                    heapq.heappush(heap, (dist[v - 1], v))
        return dist, prev

    # Dijkstra com Vetor simples (lento)
    def dijkstraVetor(self, vertice_inicial):
        n = self.quantidade_vertices_effective
        dist = [float('inf')] * n
        prev = [None] * n
        dist[vertice_inicial - 1] = 0
        S = set()
        V = set(range(1, n + 1))
        while S != V:
            u = None
            min_val = float('inf')
            for vertex in V - S:
                if dist[vertex - 1] < min_val:
                    min_val = dist[vertex - 1]
                    u = vertex
            if u is None:
                break
            S.add(u)
            for v, w in self.estruturaGrafo[u - 1]:
                if dist[v - 1] > dist[u - 1] + w:
                    dist[v - 1] = dist[u - 1] + w
                    prev[v - 1] = u
        return dist, prev

    def reconstruir_caminho(self, prev, alvo):
        caminho = []
        atual = alvo
        while atual is not None:
            caminho.append(atual)
            atual = prev[atual - 1]
        caminho.reverse()
        return caminho


# ==========================
# PARTE 1: Distâncias e Caminhos
# ==========================
arquivo = "grafo_W_1.txt"  # nome do seu arquivo .txt
grafo = Grafo(arquivo, representacao='lista')

origem = 10
alvos = [20, 30, 40, 50, 60]

dist, prev = grafo.dijkstraHeap(origem)

resultados = []
for a in alvos:
    if dist[a - 1] == float('inf'):
        resultados.append([origem, a, None, "Sem caminho"])
    else:
        caminho = grafo.reconstruir_caminho(prev, a)
        resultados.append([origem, a, round(dist[a - 1], 4), caminho])

df_caminhos = pd.DataFrame(resultados, columns=["Origem", "Destino", "Distância", "Caminho"])
print("\n==================== Caminho mínimo e distância ====================")
print(df_caminhos.to_string(index=False))

# ==========================
# PARTE 2: Tempos médios de execução
# ==========================
N = min(grafo.quantidade_vertices_effective, 2000)
grafo_teste = Grafo(arquivo, representacao='lista', limit_vertices=N)

k = 100 if N >= 100 else N
vertices_aleatorios = random.sample(range(1, N + 1), k)

# Heap
inicio = time.perf_counter()
for v in vertices_aleatorios:
    grafo_teste.dijkstraHeap(v)
fim = time.perf_counter()
tempo_heap = fim - inicio
media_heap = tempo_heap / k

# Vetor
inicio = time.perf_counter()
for v in vertices_aleatorios:
    grafo_teste.dijkstraVetor(v)
fim = time.perf_counter()
tempo_vetor = fim - inicio
media_vetor = tempo_vetor / k

df_tempos = pd.DataFrame([
    ["Dijkstra (Heap)", N, k, round(tempo_heap, 4), round(media_heap, 4)],
    ["Dijkstra (Vetor)", N, k, round(tempo_vetor, 4), round(media_vetor, 4)],
], columns=["Implementação", "Nº vértices", "k execuções", "Tempo total (s)", "Tempo médio (s)"])

print("\n==================== Tempos de execução ====================")
print(df_tempos.to_string(index=False))
