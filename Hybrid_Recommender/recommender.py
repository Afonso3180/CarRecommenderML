import numpy as np
from math import sqrt


def manhattan(carros, perfilDoUsuario):
    """Computes the Manhattan distance. """
    distance = 0
    total = 0
    for key in carros:
        if key in perfilDoUsuario:
            key = int(key)
            distance += abs(carros[key] - perfilDoUsuario[key])
            total += 1
    return distance

def cos_sim(a, b):
    """Takes 2 vectors a, b and returns the cosine similarity according
    to the definition of the dot product
    """
    comparaA = []
    comparaB = []
    for key in a:
        key = int(key)
        #arrendA=int(round(a[key]))
        #arrendB=int(round(b[key]))
        comparaA.append(a[key])
        comparaB.append(b[key])

    dot_product = np.dot(comparaA, comparaB)
    norm_a = np.linalg.norm(comparaA)
    norm_b = np.linalg.norm(comparaB)
    return dot_product / (norm_a * norm_b)

def pearson(ponto1, ponto2):
    sum_xy = 0
    sum_x = 0
    sum_y = 0
    sum_x2 = 0
    sum_y2 = 0
    n = 0
    for key in ponto1:
        if key in ponto2:
            key = int(key)
            n+=1
            x = ponto1[key]
            y = ponto2[key]
            sum_xy += x * y
            sum_x += x
            sum_y += y
            sum_x2 += x**2
            sum_y2 += y**2

    if n == 0:
        return 0

    denominator = sqrt(sum_x2 - (sum_x**2)/n) * sqrt(sum_y2 - (sum_y**2)/n)

    if (denominator == 0):
        return 0

    else:
        return (sum_xy - (sum_x * sum_y)/n)/denominator

def recommend(userName, df):
    """creates a sorted list of cars based on their distance to perfilDoUsuario"""
    distances = []
    tam = df.shape[0]
    compare = list(userName.values())
    print(compare)
    for i in range(tam):
        carro = df.iloc[i]

        #distance = pearson(list(carro)[4:], compare)
        distance = cos_sim(list(carro)[4:], compare)
        #distance = manhattan(list(carro)[4:], compare)
        distances.append((distance, list(carro)[:4]))
    # sort based on distance -- closest first
    distances.sort()
    distances.reverse() # ---- Pearson and Cosseno
    #distancesPearson = sorted(distances, key=int, reverse=True)
    #print(distances[:3])
    return distances