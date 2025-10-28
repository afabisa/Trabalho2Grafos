from collections import deque
import heapq

class Grafo:

    def __init__(self, arquivo_arestas, representacao):
        self.arestas = []
        self.representacao = representacao
        self.lerArquivo(arquivo_arestas)
        self.escolherRepresentacao(representacao)


    def lerArquivo(self, arquivo_arestas):
        with open(arquivo_arestas, "r") as file:
            tupla = (0, 0)
            soma_graus = 0
            for line in file:
                linha = line.split()
                if len(linha) == 1:
                    self.quantidade_vertices = int(linha[0])
                    self.graus = [0] * self.quantidade_vertices
                elif len(linha) == 3:
                    if float(linha[2]) < 0:
                        raise Exception("Estrutura ainda não aceita pesos negativos!")
                    tupla = (int(linha[0]), int(linha[1]), float(linha[2]))
                    self.arestas.append(tupla)
                    self.graus[int(line[0]) - 1] += 1
                    self.graus[int(line[2]) - 1] += 1
                    soma_graus += 2                   
                else:
                    tupla = (int(linha[0]), int(linha[1]))
                    self.arestas.append(tupla)
                    self.graus[int(line[0]) - 1] += 1
                    self.graus[int(line[2]) - 1] += 1
                    soma_graus += 2

        self.quantidade_arestas = len(self.arestas)
        self.graus.sort()
        self.grau_max = self.graus[-1]
        self.grau_min = self.graus[0]
        self.grau_medio = soma_graus / self.quantidade_vertices
        self.mediana_grau = self.graus[int(self.quantidade_vertices / 2) - 1]

        with open("saida.txt", "w") as f:
            f.write(f"Numero de vertices: {self.quantidade_vertices}\n")
            f.write(f"Numero de arestas: {self.quantidade_arestas}\n")
            f.write(f"Grau minimo: {self.grau_min}\n")
            f.write(f"Grau Maximo: {self.grau_max}\n")
            f.write(f"Grau Medio: {self.grau_medio}\n")
            f.write(f"Mediana de grau: {self.mediana_grau}\n")
            
    
    def criarListaAdjacencia(self):
        lista_adjacencia = [deque() for _ in range(self.quantidade_vertices)]
        if len(self.arestas[0]) == 2: 
            for u, v in self.arestas:
                lista_adjacencia[u - 1].append(v)
                lista_adjacencia[v - 1].append(u)
            return lista_adjacencia
        else:
            for u, v, x in self.arestas:
                lista_adjacencia[u - 1].append((v, x))
                lista_adjacencia[v - 1].append((u, x))
            return lista_adjacencia            

    def criarMatrizAdjacencia(self):
        matriz_adjacencia = [[0] * self.quantidade_vertices for _ in range(self.quantidade_vertices)]
        if len(self.arestas[0]) == 2:
            for u, v in self.arestas:
                matriz_adjacencia[u - 1][v - 1] = 1
                matriz_adjacencia[v - 1][u - 1] = 1
            return matriz_adjacencia
        else:
            for u, v, x in self.arestas:
                matriz_adjacencia[u - 1][v - 1] = x
                matriz_adjacencia[v - 1][u - 1] = x
            return matriz_adjacencia

    def escolherRepresentacao(self, representacao):
        if representacao == 'matriz':
            self.estruturaGrafo = self.criarMatrizAdjacencia()
        elif representacao == 'lista':
            self.estruturaGrafo = self.criarListaAdjacencia()
        else:
            raise Exception('Representação inválida')


    def BFS(self, vertice_inicial):
        pai = [0] * self.quantidade_vertices
        nivel = [0] * self.quantidade_vertices
        fila = deque()
        descobertos = [0] * self.quantidade_vertices
        descobertos[vertice_inicial - 1] = 1
        fila.append(vertice_inicial)
        pai[vertice_inicial - 1] = 0
        nivel[vertice_inicial - 1] = 0
        if self.representacao == 'matriz':
            while fila:
                v = fila.popleft()
                for w, valor in enumerate(self.estruturaGrafo[v - 1]):
                    if valor == 1:
                        if descobertos[w - 1] == 0:
                            descobertos[w - 1] = 1
                            fila.append(w)
                            pai[w - 1] = v
                            nivel[w - 1] = nivel[v - 1] + 1

        else:
            while fila:
                v = fila.popleft()
                for w in self.estruturaGrafo[v - 1]:
                    if descobertos[w - 1] == 0:
                        descobertos[w - 1] = 1
                        fila.append(w)
                        pai[w - 1] = v
                        nivel[w - 1] = nivel[v - 1] + 1
        return pai, nivel

    def DFS(self, vertice_inicial):
        pai = [0] * self.quantidade_vertices
        nivel = [0] * self.quantidade_vertices
        pilha = deque()
        marcado = [0] * self.quantidade_vertices
        pilha.append(vertice_inicial)
        pai[vertice_inicial - 1] = 0
        nivel[vertice_inicial - 1] = 0
        if self.representacao == 'matriz':
            while pilha:
                u = pilha.pop()
                if marcado[u - 1] == 0:
                    marcado[u - 1] = 1
                    for v, valor in enumerate(self.estruturaGrafo[u - 1]):
                        if valor == 1:
                            pilha.append(v)

        else:
            while pilha:
                u = pilha.pop()
                if marcado[u - 1] == 0:
                    marcado[u - 1] = 1
                    for v in self.estruturaGrafo[u - 1]:
                        pilha.append(v)  
        
        return pai, nivel

    def distanciaDiametro(self, primeiro, segundo): 
        pai, nivel = self.BFS(primeiro)
        diametro = max(nivel)
        return nivel[segundo - 1], diametro
    
    def componenetesConexos(self):
        visitados = [False] * self.quantidade_vertices
        qtd_componentes = 0
        componentes = []
        tamanhos = []
        fila = deque()
        iterador = 0
        for node in range(self.quantidade_vertices):
            if not visitados[node]:
                qtd_componentes += 1
                componentes.append([])
                tamanhos.append([])
                fila.append(node)
                visitados[node] = True
                componentes[iterador].append(node)
                tamanhos[iterador] += 1
                
                while fila:
                    atual = fila.popleft()
                    for vizinho in self.estruturaGrafo[atual - 1]:
                        if not visitados[vizinho]:
                            fila.append(vizinho)
                            visitados[vizinho] = True
                            componentes[iterador].append(vizinho)
                            tamanhos[iterador] += 1
        
        componentes.sort(key=lambda a: len(a), reverse=True)
        tamanhos.sort()
        tamanhos.reverse()
        return qtd_componentes, tamanhos, componentes
    
    def dijkstraVetor(self, vertice_incial):
        distancias = [float('inf') for _ in range(self.quantidade_vertices)]
        conjunto_S = set()
        conjunto_V = {vertice + 1 for vertice in range(self.quantidade_vertices)}
        distancias[vertice_incial - 1] = 0
        anteriores = [None for _ in range(self.quantidade_vertices)]
        while conjunto_S != conjunto_V:
            u = next(iter(conjunto_V.difference(conjunto_S)))
            for vertice in conjunto_V.difference(conjunto_S):
                if distancias[vertice - 1] < distancias[u - 1]:
                    u = vertice
            conjunto_S.add(u)
            if self.representacao == 'lista':
                for vizinho, peso in self.estruturaGrafo[u - 1]:
                    if distancias[vizinho - 1] > distancias[u - 1] + peso:
                        distancias[vizinho - 1] = distancias[u - 1] + peso
                        anteriores[vizinho - 1] = u
            else:
                for vizinho, peso in enumerate(self.estruturaGrafo[u - 1]):
                    if peso == 0:
                        continue   
                    if distancias[vizinho] > distancias[u - 1] + peso:
                        distancias[vizinho] = distancias[u - 1] + peso
                        anteriores[vizinho] = u
        return distancias, anteriores
    
    def dijkstraHeap(self, vertice_incial):
        distancias = [float('inf') for vertice in range(self.quantidade_vertices)]
        distancias[vertice_incial - 1] = 0
        anteriores = [None for _ in range(self.quantidade_vertices)]
        heap = [(0, vertice_incial)]
        
        while heap:
            
            distancia_atual, vertice_atual = heapq.heappop(heap)

            if distancia_atual > distancias[vertice_atual - 1]:
                continue
            
            if self.representacao == 'lista':
                for vizinho, peso in self.estruturaGrafo[vertice_atual - 1]:
                    distancia_nova = distancia_atual + peso
                    if distancia_nova < distancias[vizinho - 1]:
                        distancias[vizinho - 1] = distancia_nova
                        anteriores[vizinho - 1] = vertice_atual
                        heapq.heappush(heap, (distancia_nova, vizinho))
            else:
                for vizinho, peso in enumerate(self.estruturaGrafo[vertice_atual - 1]):
                    if peso == 0:
                        continue
                    distancia_nova = distancia_atual + peso
                    if distancia_nova < distancias[vizinho]:
                        distancias[vizinho] = distancia_nova
                        anteriores[vizinho] = vertice_atual
                        heapq.heappush(heap, (distancia_nova, vizinho + 1))
        return distancias, anteriores

x = Grafo('exemplo.txt', 'matriz')

print(x.dijkstraHeap(4)) 