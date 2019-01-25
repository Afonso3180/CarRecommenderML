
from math import sqrt
import numpy as np
from pandas import Series, DataFrame
import pandas as pd
import csv
#from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import mode
from sklearn.metrics import average_precision_score

def precisao(relevantes, numrecomendados):
    # Vai considerar apenas numeros
    if relevantes <= 0:
        return 0
    else:
        relevantes = float(relevantes)
        preci = (relevantes/numrecomendados)
        return preci

#recall avalia a fracao de itens relevantes que foram recomendados
def recall(relevantes, selected):
    # Vai considerar apenas numeros
    if relevantes <= 0:
        return 0
    else:
        relevantes = float(relevantes)
        rec = (relevantes/selected)
        return rec

# Ccombina os valores da precisao e do recall em uma unica metrica
def F_measure(relevantes, numrecomendados, selected):
    pre = precisao(relevantes, numrecomendados)
    re = recall(relevantes, selected)

    if pre <= 0 and re <= 0:
        return 0
    else:
        Fmeasure = (2*pre*re)/(pre+re)
        return Fmeasure

def armazenarNotas(pre,re,Fme):
    with open('C:/Users/User Acer/Desktop/Paic/Datasets/Avaliacao/avaliacoes.csv', 'a') as f:
        writer = csv.writer(f)
        pre = float("{0:.3f}".format(pre))
        re = float("{0:.3f}".format(re))
        Fme = float("{0:.3f}".format(Fme))

        writer.writerow([pre,re,Fme])

#///////////////////////////////////////////// PROCEDIMENTOS COM A AVALIACAO ////////////////////////////////////////////////////////////////////////////////

#sera considerado 3 como numrecomendados
def avaliar(selected):
    relevantes = int(input("Sera medida a precisao, digitar a quantidade de automoveis relevantes e tecle 'Enter': "))

    if relevantes <= 3 and relevantes >=0:

        pre = precisao(relevantes, 3)
        #print(pre)
        re = recall(relevantes, selected)
        #print(re)
        Fme = F_measure(relevantes, 3, selected)
        #print(Fme)

        armazenarNotas(pre,re,Fme)

        print("a precisao e: %1.4f" %pre)
        print("o recall e: %1.4f" %re)
        print("Fmeasure e: %1.4f" %Fme)

    else:

        print("Valor invalido, tente novamente: ")
        avaliar(selected)

#//////////////////////////////////////////////////AVALIAR SISTEMA///////////////////////////////////////////////////////////////////////////////////////////
#TRABALHA COM LEITURA DO DATASET AVALIACAO E CALCULA A MEDIA DAS AVALIACOES

def avaliasistema():
    avalia = pd.read_csv("C:/Users/User Acer/Desktop/Paic/Datasets/Avaliacao/avaliacoes.csv")
    avalia_dataset = pd.DataFrame(avalia, columns=['pre','re','Fme'])

    soma = sum(avalia_dataset["Fme"].values)
    media = soma/(len(avalia_dataset["Fme"].values))

    return(media)
