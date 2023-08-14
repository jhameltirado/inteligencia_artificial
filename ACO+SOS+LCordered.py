#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 13 20:15:31 2023

@author: jhameltb
"""

import math
import random
import numpy as np
import matplotlib.pyplot as plt
import time

inicio = time.time()

# Calcular distancias nuevo camino
def calculate_dis(dis, path):
    
    length = 0
    for i in range(len(path) - 1):
        length += dis[path[i]][path[i + 1]]
    return length

# Selección de siguiente ciudad
def roulette(pooling):
    sum_num = sum(pooling)
    temp_num = random.random() # se obtiene un número aleatorio
    probability = 0
    
    # se recorren los cálculos por ciudad candidata
    for i in range(len(pooling)): 
        # se suman las probabilidades
        probability += pooling[i] / sum_num
        # si la probabilidad es mayor al número aleatorio, se selecciona el actual
        if probability >= temp_num:
            return i
    return len(pooling)

# Construir camino
def define_path(dis, pheromone, alpha, beta, ant):
    
    # comenzamos el camino en la ciudad asignada a la hormiga
    path = [ant]
    curr_node = ant
    candidate_cities = []
    
    # copiamos la lista que contiene las distancias de la ciudad actual a las demás ciudades
    ord_dis = []
    ord_dis = dis[curr_node].copy()
    # ordenamos dicha lista de manera ascendente
    ord_dis.sort()
    
    for i in range(1,len(dis)):
        # obtenemos el índice correspondiente a la ciudad con dicha distancia
        cit = dis[curr_node].index(ord_dis[i])
        
        # agregamos la ciudad a la lista de ciudades candidatas
        candidate_cities.append(cit)

    for i in range(len(dis) - 1):
        
        roulette_pooling = []
        
        for city in candidate_cities:

            # lista del producto de las feromonas por la visibilidad del camino a cada ciudad  candidata desde la ciudad actual
            roulette_pooling.append(math.pow(pheromone[curr_node][city], alpha) * math.pow(1/dis[curr_node][city], beta))
            

        # se obtiene el índice que representa la siguiente ciudad seleccionada
        index = roulette(roulette_pooling)
        curr_node = candidate_cities[index] # se vuelve nuestra ciudad actual
        path.append(curr_node) # se agrega la ciudad al camino
        candidate_cities.pop(index) # se elimina de las ciudades candidatas
    
    # se agrega 0 porque vuelve a la ciudad de inicio
    path.append(ant)
    return path

#==== STP CON COLONIA DE HORMIGAS IMPLEMENTANDO LISTA DE CANDIDATOS Y ALGORITMO SOS

# Definición de parámetros
alpha = 1
beta = 5
rho = 0.1
Q = 1
ants = 30
itera = 200
city_num = 100

min_coord = 0
max_coord = 100

# LO SIGUIENTE SE DESCOMENTA CUANDO SE NECESITA 

# ================ Generación de coordenadas aleatorias =======================
coord_x = [random.uniform(min_coord, max_coord) for _ in range(city_num)]
coord_y = [random.uniform(min_coord, max_coord) for _ in range(city_num)]

# Almacenamiento de coordenadas para probar con las mismas ciudades
np.savetxt('coord_x.txt', coord_x, fmt='%f')
np.savetxt('coord_y.txt', coord_y, fmt='%f')
# =============================================================================

# Cargar lo almacenado para probar con mismo problema pero diferentes parámetros
# coord_x = np.loadtxt('coord_x.txt', dtype=float)
# coord_y = np.loadtxt('coord_y.txt', dtype=float)

# Definir distancias y fermonas iniciales
dis = [[0 for i in range(city_num)] for i in range(city_num)]  # se inicializa la matriz de distancias
for i in range(city_num):
    
    for j in range(i, city_num):
        
        # Se llena la matriz calculando las distancias entre ciudades
        temp_dis = math.sqrt((coord_x[i] - coord_x[j]) ** 2 + (coord_y[i] - coord_y[j]) ** 2)
        dis[i][j] = temp_dis
        dis[j][i] = temp_dis
        
pheromone = [[1 for _ in range(city_num)] for _ in range(city_num)] # inicializamos feromonas en 1
itera_best = []  # el camino más corto de cada iteracion
best_path = []
best_length = 1e6 # definimos distancia alta para la futura comparación


# Iteraciones
for _ in range(itera):

    # Construcción de soluciones
    ant_path = [] # camino por hormiga
    ant_path_length = [] # distancia por hormiga 
    
    for i in range(ants):
        
        # se define el camino con base en feromonas, visibilidad y probabilidad
        new_path = define_path(dis, pheromone, alpha, beta, i)
        
        # se calcula la distancia total del camino
        new_length = calculate_dis(dis, new_path)
        
        # se agrega el camino a la lista de caminos por hormiga
        ant_path.append(new_path)
        # se agrega la distancia total del camino a la lista de distancias totales
        ant_path_length.append(new_length)
    
    # se almacena el camino con la menor distancia de la iteración actual
    iter_best_path_length = min(ant_path_length)
    
    # si la distancia del camino más corto de la iteración actual es menor que el camino más corto global
    if iter_best_path_length < best_length:
        
        # se cambia el mejor camino global
        best_length = iter_best_path_length
        
        # se determina como mejor camino el mejor de la iteración
        best_path = ant_path[ant_path_length.index(iter_best_path_length)]
        
    # guardamos el mejor camino global actual (para análisis de convergencia)
    itera_best.append(best_length)
    
    #APLICAR ALGORITMO SOS 
    for i in range(ants):
            # Comparar solución de la hormiga actual con soluciones de las hormigas vecinas.
            for j in range(ants):

                if j != i and ant_path_length[j] < ant_path_length[i]:
                    current_path = ant_path[i]
                    neighbor_path = ant_path[j]
                    
                    # Encontrar qué parte del recorrido difieren
                    diff_cities = [(x, y) for x, y in zip(current_path, neighbor_path) if x != y]
                    
                    # Intercambiar dichas ciudades en las que difieren
                    for x, y in diff_cities:
                        current_path[current_path.index(x)] = y
                        current_path[current_path.index(y)] = x
                        
                      # Actualizar feromonas en el camino modificado (reforzando)
                    delta = Q / ant_path_length[i]
                    for k in range(city_num):
                        pheromone[current_path[k]][current_path[k + 1]] += delta
                        pheromone[current_path[k + 1]][current_path[k]] += delta
                    
                    # Reemplazar el camino antiguo con el camino modificado en la lista de ant_path
                    ant_path[i] = current_path
                    
                    # Actualizar best_length si se encontró un camino más corto
                    if ant_path_length[j] < best_length:
                        best_length = ant_path_length[j]
                        best_path = ant_path[j].copy()  # Actualizar el mejor camino encontrado

    # Evaporacion de feromonas
    for i in range(city_num):
        for j in range(i, city_num):
            pheromone[i][j] *= (1 - rho)
            pheromone[j][i] *= (1 - rho)
            
fin = time.time()
print("Tiempo de ejecución:")
print(fin-inicio)
    
print('\nEl mejor camino es: ', best_path)
print('\nSu distancia es de: ', best_length)
    
# Resultados

x = [i for i in range(itera)]

plt.figure()
plt.plot(x, itera_best, linewidth=2, color='green')
plt.xlabel("Iteraciones")
plt.ylabel("Valores Óptimos globales")
plt.show()

plt.figure()

plt.scatter(coord_x, coord_y, color='#00008B')

for i in range(len(best_path) - 1):
    temp_x = [coord_x[best_path[i]], coord_x[best_path[i + 1]]]
    temp_y = [coord_y[best_path[i]], coord_y[best_path[i + 1]]]
    plt.plot(temp_x, temp_y, color='#8FBC8F')

plt.show()
