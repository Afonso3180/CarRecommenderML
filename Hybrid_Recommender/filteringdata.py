#


#  Para testar digite recommend('Bonifacio', dataset)
import random
from math import sqrt
from collections import OrderedDict
import time
import numpy as np
from pandas import Series, DataFrame
import pandas as pd
import csv
#from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import mode
import recommender as rec
import avaliar as ava
from sklearn.metrics import average_precision_score
#importando dataset e nomeando como dataset


#///////////////////////////////////////////// DATASET - CARREGANDO E TRATANDO //////////////////////////////////////////////////////////////////////////////

dataset = pd.read_csv("C:/Users/User Acer/Desktop/Paic/Datasets/DatasetCarrosFiltConteudo.csv")
df = pd.DataFrame(dataset, columns=['Marca','Modelo','Ano','Concessionaria','Conforto','Seguranca','GastosFixos','Desempenho','Carroceria','NumLugares','NumeroDePortas','Finalidade','Combustivel','Valor','SensorDeEstacionamento','ArCondicionado','Bluetooth','Direcao','BancoCouro','GPS','VidrosEletricos','PilotoAutomatico','TetoSolar','TamPortaMala','Cacamba','ComputadorBordo','DisponibiliPeca','Airbag','ABS','Blindagem','FarolNeblina','IPVA','ConsumoGasolina','Seguro','Manutencao','Motor','CavalosForca','VelMaxima'])

df['Conforto'] = df[['SensorDeEstacionamento','ArCondicionado','Bluetooth','Direcao','BancoCouro','GPS','VidrosEletricos','PilotoAutomatico','TetoSolar','TamPortaMala','Cacamba','ComputadorBordo']].mean(axis=1)
df['Seguranca'] = df[['DisponibiliPeca','Airbag','ABS','Blindagem','FarolNeblina']].mean(axis=1)
df['GastosFixos'] = df[['IPVA','ConsumoGasolina','Seguro','Manutencao']].mean(axis=1)
df['Desempenho'] = df[['IPVA','ConsumoGasolina','Seguro','Manutencao']].mean(axis=1)

df.drop(['SensorDeEstacionamento','ArCondicionado','Bluetooth','Direcao','BancoCouro','GPS','VidrosEletricos','PilotoAutomatico','TetoSolar','TamPortaMala','Cacamba','ComputadorBordo','DisponibiliPeca','Airbag','ABS','Blindagem','FarolNeblina','IPVA','ConsumoGasolina','Seguro','Manutencao','Motor','CavalosForca','VelMaxima'], axis=1, inplace=True)

training_dataset = pd.DataFrame(dataset, columns=['Marca','Modelo','Ano','Concessionaria','Conforto','Seguranca','GastosFixos','Desempenho','Carroceria','NumLugares','NumeroDePortas','Finalidade','Combustivel','Valor','SensorDeEstacionamento','ArCondicionado','Bluetooth','Direcao','BancoCouro','GPS','VidrosEletricos','PilotoAutomatico','TetoSolar','TamPortaMala','Cacamba','ComputadorBordo','DisponibiliPeca','Airbag','ABS','Blindagem','FarolNeblina','IPVA','ConsumoGasolina','Seguro','Manutencao','Motor','CavalosForca','VelMaxima'])
training_dataset.drop(['Concessionaria','Conforto','Seguranca','GastosFixos','Desempenho','Carroceria','NumLugares','NumeroDePortas','Finalidade','Combustivel','Valor','SensorDeEstacionamento','ArCondicionado','Bluetooth','Direcao','BancoCouro','GPS','VidrosEletricos','PilotoAutomatico','TetoSolar','TamPortaMala','Cacamba','ComputadorBordo','DisponibiliPeca','Airbag','ABS','Blindagem','FarolNeblina','IPVA','ConsumoGasolina','Seguro','Manutencao','Motor','CavalosForca','VelMaxima'], axis=1, inplace=True)

#/////////////////////////////////////////////////////////// PERFIL DO USUARIO //////////////////////////////////////////////////////////////////////////////

#  Perfil do usuario
key = ["Conforto","Seguranca","GastosFixos","Desempenho","Carroceria","NumLugares","NumeroDePortas","Finalidade","Combustivel","Valor"]
#perfilUsuario = {"Conforto": 2,"Seguranca": 2,"GastosFixos": 1,"Desempenho": 1,"Carroceria": 1.5,"NumLugares": 5,"NumeroDePortas": 3,"Finalidade": 1,"Combustivel": 2,"Valor": 4}
perfilUsuario = {}
# Preencher o perfil usuario
user = []
def preencheperfil():
    print("O Perfil do usuario deve ser preenchido a seguir com valores numericos de 1 a 5: ")
    perfilUsuario['Conforto'] = int(input('Conforto: '))
    perfilUsuario['Seguranca'] = int(input('Seguranca: '))
    perfilUsuario['GastosFixos'] = int(input('Gastos Fixos: '))
    perfilUsuario['Desempenho'] = int(input('Desempenho: '))
    perfilUsuario['Carroceria'] = float(input('Carroceria: '))
    perfilUsuario['NumLugares'] = int(input('Numero de Lugares: '))
    perfilUsuario['NumeroDePortas'] = int(input('Numero de Portas: '))
    perfilUsuario['Finalidade'] = int(input('Finalidade: '))
    perfilUsuario['Combustivel'] = int(input('Combustivel: '))
    perfilUsuario['Valor'] = int(input('Valor: '))

    user = []
    for j in key: #O perfil do usuario recebe os valores inseridos
        user.append(perfilUsuario[j])




#///////////////////////////////////////////////////SEPARA O DATASET DE FORMA ALEATORIA E SALVA EM UM CSV///////////////////////////////////////////////////
#Seleciona Carros Aleatorios
'''
for x in range(75):
    rand = random.randint(0,(training_dataset.shape[0])-1)
    training_dataset.drop(training_dataset.index[rand], axis=0, inplace=True)

#Sera ececutado uma vez e salvo em um arquivo
training_dataset['selected'] = pd.Series(0, index=training_dataset.index)
training_dataset.reset_index(drop=True, inplace=True)

training_dataset.to_csv('training dataset', index=True)

print(training_dataset)
print("\n")
'''
#     FOI GERADO E SALVO UMA SO VEZ
#     NAO APAGAR EM HIPOTESE ALGUMA
#//////////////////////////////////////////////////GERA A SELECAO PARA COMPARACAO DOS AUTOMOVEIS DO DATASET DE TREINO///////////////////////////////////////

train_dataset = pd.read_csv("C:/Users/User Acer/Desktop/Paic/Datasets/training_dataset.csv")
training_dataset = pd.DataFrame(train_dataset, columns=['ID','Marca','Modelo','Ano','selected'])
training_dataset.drop(['ID', 'selected'],axis=1, inplace=True)
print(training_dataset)

selected = [] #Vai ser preenchida com os carros selecionados
selected_numbers = 0
IDS = [] #Vai ser preenchida com os carros presentes na lista " selected "
entrada = 1
print("Quais carros lhe interessam? digitar o numero e selecionar 'Enter', ao terminar digitar e selecionar '-1'")
while(entrada != -1): #Vai preencher uma lista com os carros selecionados
    entrada = int(input("ID do Carro: "))
    if entrada != -1:
        if entrada in IDS:
            print("Carro ja selecionado")
        else:
            selected.append(list(training_dataset.iloc[entrada]))
            IDS.append(entrada)

tam = df.shape[0]
user_selected = []
for car in selected: #Vai preencher uma lista com os carros selecionados junto dos respectivos atributos
   #print(car)
    for i in range(tam - 1):
        carro = df.iloc[i]
        if ((list(carro)[:11])[1] == car[1]):
            user_selected.append(list(carro))

selected_numbers = len(selected)
print(selected_numbers)

# RETORNA A LISTA "user_selected"

#//////////////////////////////////////////////////////RACIOCINIO BASEADO EM CASOS///////////////////////////////////////////////////////////////////////////

#No pacote CaseBased.py

#///////////////////////////////////////////////////////  MAIN  /////////////////////////////////////////////////////////////////////////////////////////////

preencheperfil()

rec.recommend(perfilUsuario, df)
ava.avaliar(selected_numbers)
#print(ava.avaliasistema())
