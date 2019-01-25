import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.datasets import make_circles
from pandas import Series, DataFrame
import pandas as pd
import csv
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import mode
import recommender as rec
import avaliar as ava
from sklearn.metrics import average_precision_score
import pickle

# Create the dataset

# Dataset for content based filtering
dataset = pd.read_csv("C:/Users/User Acer/Desktop/Paic/Datasets/DatasetCarrosFiltConteudo.csv")
df = pd.DataFrame(dataset, columns=['Marca','Modelo','Ano','Concessionaria','Conforto','Seguranca','GastosFixos','Desempenho','Carroceria','NumLugares','NumeroDePortas','Finalidade','Combustivel','Valor','SensorDeEstacionamento','ArCondicionado','Bluetooth','Direcao','BancoCouro','GPS','VidrosEletricos','PilotoAutomatico','TetoSolar','TamPortaMala','Cacamba','ComputadorBordo','DisponibiliPeca','Airbag','ABS','Blindagem','FarolNeblina','IPVA','ConsumoGasolina','Seguro','Manutencao','Motor','CavalosForca','VelMaxima'])

#perfilUsuario = {"Conforto": 2,"Seguranca": 2,"GastosFixos": 1,"Desempenho": 1,"Carroceria": 1.5,"NumLugares": 5,"NumeroDePortas": 3,"Finalidade": 1,"Combustivel": 2,"Valor": 4}

df['Conforto'] = df[['SensorDeEstacionamento','ArCondicionado','Bluetooth','Direcao','BancoCouro','GPS','VidrosEletricos','PilotoAutomatico','TetoSolar','TamPortaMala','Cacamba','ComputadorBordo']].mean(axis=1)
df['Seguranca'] = df[['DisponibiliPeca','Airbag','ABS','Blindagem','FarolNeblina']].mean(axis=1)
df['GastosFixos'] = df[['IPVA','ConsumoGasolina','Seguro','Manutencao']].mean(axis=1)
df['Desempenho'] = df[['IPVA','ConsumoGasolina','Seguro','Manutencao']].mean(axis=1)

df.drop(['SensorDeEstacionamento','ArCondicionado','Bluetooth','Direcao','BancoCouro','GPS','VidrosEletricos','PilotoAutomatico','TetoSolar','TamPortaMala','Cacamba','ComputadorBordo','DisponibiliPeca','Airbag','ABS','Blindagem','FarolNeblina','IPVA','ConsumoGasolina','Seguro','Manutencao','Motor','CavalosForca','VelMaxima'], axis=1, inplace=True)

training_dataset = pd.DataFrame(dataset, columns=['Marca','Modelo','Ano','Concessionaria','Conforto','Seguranca','GastosFixos','Desempenho','Carroceria','NumLugares','NumeroDePortas','Finalidade','Combustivel','Valor','SensorDeEstacionamento','ArCondicionado','Bluetooth','Direcao','BancoCouro','GPS','VidrosEletricos','PilotoAutomatico','TetoSolar','TamPortaMala','Cacamba','ComputadorBordo','DisponibiliPeca','Airbag','ABS','Blindagem','FarolNeblina','IPVA','ConsumoGasolina','Seguro','Manutencao','Motor','CavalosForca','VelMaxima'])
training_dataset.drop(['Concessionaria','Conforto','Seguranca','GastosFixos','Desempenho','Carroceria','NumLugares','NumeroDePortas','Finalidade','Combustivel','Valor','SensorDeEstacionamento','ArCondicionado','Bluetooth','Direcao','BancoCouro','GPS','VidrosEletricos','PilotoAutomatico','TetoSolar','TamPortaMala','Cacamba','ComputadorBordo','DisponibiliPeca','Airbag','ABS','Blindagem','FarolNeblina','IPVA','ConsumoGasolina','Seguro','Manutencao','Motor','CavalosForca','VelMaxima'], axis=1, inplace=True)

#Dataset for content based filtering

datasetML = pd.read_csv("C:/Users/User Acer/Documents/GitHub/CarRecommenderML/Datasets/SVM/DatasetCarrosComClasse.csv")
dfML = pd.DataFrame(datasetML, columns=['Marca','Modelo','Ano','Conforto','Seguranca','GastosFixos','Desempenho','Carroceria','NumLugares','NumeroDePortas','Finalidade','Combustivel','Valor','Valido'])

toSVM = dfML[['Conforto','Seguranca','GastosFixos','Desempenho','Carroceria','NumLugares','NumeroDePortas','Finalidade','Combustivel','Valor','Valido']]

carrosAExibir = df[['Marca','Modelo','Ano']]
#                               Faz Recomendacoes para alimentar um dataset que sera utilizado no modelo de machine learning
key = ["Conforto","Seguranca","GastosFixos","Desempenho","Carroceria","NumLugares","NumeroDePortas","Finalidade","Combustivel","Valor"]
perfilUsuario = {"Conforto": 2,"Seguranca": 2,"GastosFixos": 1,"Desempenho": 1,"Carroceria": 1.5,"NumLugares": 5,"NumeroDePortas": 3,"Finalidade": 1,"Combustivel": 2,"Valor": 4}
#perfilUsuario = {}
user = []

IDs = []

# Preencher o perfil usuario
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

    for j in key: #O perfil do usuario recebe os valores inseridos
        user.append(perfilUsuario[j])

#Cria o dataset que vai ser utilizado no Aprendizado

carros = [] #Criado com o intuito de facilitar ao armazenar no dataset
def adicionaItens():
    #Exibe os carros do Dataset
    cont = 0
    tam = df.shape[0]
    #compare = list(userName.values())
    for i in range(tam):
        car = carrosAExibir.iloc[i]
        print(cont, list(car))
        carros.append(list(car))
        cont+=1

#Avalia e adiciona no Dataset

    #Avaliacao das recomendacoes validas
    print("digite o valor referente ao index de cada item valido e clique enter, ao finalizar digitar e dar entem em -1: ")
    indexvalidos = []
    entrada = 0
    while(entrada != -1):
        print("index: ")
        entrada = int(input())

        if (entrada != -1):
            indexvalidos.append(entrada)
    #print(indexvalidos) Utilizado apenas se necessario debug

    #formata as linhas para serem adicionadas no dataset
    linhas = []
    for i in indexvalidos:
        linhas.append(list(carros[i]))
    for i in range(len(linhas)):
        for j in range(len(user)):
            linhas[i].append(user[j])
        linhas[i].append(1)

    # As recomendacoes consideradas validas estao prontas para serem escritas no dataset

    with open('C:/Users/User Acer/Documents/GitHub/CarRecommenderML/Datasets/SVM/DatasetCarrosComClasse.csv', 'a') as f:
        writer = csv.writer(f)
        for i in range(len(linhas)):
            writer.writerow(linhas[i])

    # Insere os itens nao validos
    print("digite o valor referente ao index de cada item nao valido e clique enter, ao finalizar digitar e dar entem em -1: ")
    naovalidos = []
    entradainvalidos = 0
    while(entradainvalidos != -1):
        print("index: ")
        entradainvalidos = int(input())

        if (entradainvalidos != -1):
            naovalidos.append(entradainvalidos)
    #print(naovalidos) Utilizado apenas se necessario debug

    linhasinvalidas = []

    for i in naovalidos:
        linhasinvalidas.append(list(carros[i]))
    for i in range(len(linhasinvalidas)):
        for j in range(len(user)):
            linhasinvalidas[i].append(user[j])
        linhasinvalidas[i].append(0)

    with open('C:/Users/User Acer/Documents/GitHub/CarRecommenderML/Datasets/SVM/DatasetCarrosComClasse.csv', 'a') as f:
        writer = csv.writer(f)
        for i in range(len(linhasinvalidas)):
            writer.writerow(linhasinvalidas[i])



#                                Filtragem Baseada em Conhecimento

def SVM():
    X = toSVM.drop('Valido', axis=1)
    Y = toSVM['Valido']

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.20)


    #svclassifier = SVC(kernel='poly')
    #svclassifier.fit(X_train, y_train)

    #pickle.dump(svclassifier, open('modelo.sav', 'wb'))

    svclassifier = pickle.load(open('modelo.sav', 'rb'))


    y_pred = svclassifier.predict(X_test)

    print(confusion_matrix(y_test,y_pred))
    print(classification_report(y_test,y_pred))

    to_get_rec = (X_test.index.tolist())

    to_compare = list(y_pred)

    for i in range(len(X_test.index.tolist())):
        if to_compare[i] == 1:
            IDs.append(to_get_rec[i])

    print(IDs)


#                               Filtragem Baseada em Conteudo

def reco():
    lista = rec.recommend(perfilUsuario, df)
    cont = 0
    for i in lista:
        if cont <= 3:
            toList = list(i)
            torec = toList[1]
            lista_conteudo.append(torec[:3])
            cont +=1
        else:
            pass

def recML(IDs):

    for i in IDs:
        value = (df.iloc[i])
        recommendML.append((list(value))[:3])

lista_conteudo = []
recommendML = []
#preencheperfil()
#adicionaItens()
SVM()
recML(IDs)
reco()

recomendacoesHibrida = []
for i in lista_conteudo:
    if i in recommendML:
        pass
    else:
        recomendacoesHibrida.append(i)

recomendacoesHibrida = recomendacoesHibrida + recommendML
print(recomendacoesHibrida)

#print(y_pred)





