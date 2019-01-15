import sklearn
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.model_selection import train_test_split
from pandas import Series, DataFrame
import pandas as pd
import collections
import recommender as rec
import csv

#Dataset Original
#                               Importa e Trabalha no Dataset Original de Machine Learning

dataset = pd.read_csv("C:/Users/User Acer/Documents/GitHub/CarRecommenderML/Datasets/NaiveBayes/DatasetCarros.csv")
df = pd.DataFrame(dataset, columns=['Marca','Modelo','Ano','Concessionaria','Conforto','Seguranca','GastosFixos','Desempenho','Carroceria','NumLugares','NumeroDePortas','Finalidade','Combustivel','Valor','SensorDeEstacionamento','ArCondicionado','Bluetooth','Direcao','BancoCouro','GPS','VidrosEletricos','PilotoAutomatico','TetoSolar','TamPortaMala','Cacamba','ComputadorBordo','DisponibiliPeca','Airbag','ABS','Blindagem','FarolNeblina','IPVA','ConsumoGasolina','Seguro','Manutencao','Motor','CavalosForca','VelMaxima'])

df['Conforto'] = df[['SensorDeEstacionamento','ArCondicionado','Bluetooth','Direcao','BancoCouro','GPS','VidrosEletricos','PilotoAutomatico','TetoSolar','TamPortaMala','Cacamba','ComputadorBordo']].mean(axis=1)
df['Seguranca'] = df[['DisponibiliPeca','Airbag','ABS','Blindagem','FarolNeblina']].mean(axis=1)
df['GastosFixos'] = df[['IPVA','ConsumoGasolina','Seguro','Manutencao']].mean(axis=1)
df['Desempenho'] = df[['IPVA','ConsumoGasolina','Seguro','Manutencao']].mean(axis=1)

df.drop(['SensorDeEstacionamento','ArCondicionado','Bluetooth','Direcao','BancoCouro','GPS','VidrosEletricos','PilotoAutomatico','TetoSolar','TamPortaMala','Cacamba','ComputadorBordo','DisponibiliPeca','Airbag','ABS','Blindagem','FarolNeblina','IPVA','ConsumoGasolina','Seguro','Manutencao','Motor','CavalosForca','VelMaxima'], axis=1, inplace=True)

carrosAExibir = df[['Marca','Modelo','Ano']]
#Recomendacoes e Novo Dataset
#                               Faz Recomendacoes para alimentar um dataset que sera utilizado no modelo de machine learning
key = ["Conforto","Seguranca","GastosFixos","Desempenho","Carroceria","NumLugares","NumeroDePortas","Finalidade","Combustivel","Valor"]
#perfilUsuario = {"Conforto": 2,"Seguranca": 2,"GastosFixos": 1,"Desempenho": 1,"Carroceria": 1.5,"NumLugares": 5,"NumeroDePortas": 3,"Finalidade": 1,"Combustivel": 2,"Valor": 4}
perfilUsuario = {}
# Preencher o perfil usuario
print("O Perfil do usuario deve ser preenchido a seguir com valores numericos de 1 a 5: ")
perfilUsuario['Conforto'] = int(input('Conforto: '))
perfilUsuario['Seguranca'] = int(input('Seguranca: '))
perfilUsuario['GastosFixos'] = int(input('Gastos Fixos: '))
perfilUsuario['Desempenho'] = int(input('Desempenho: '))
perfilUsuario['Carroceria'] = int(input('Carroceria: '))
perfilUsuario['NumLugares'] = int(input('Numero de Lugares: '))
perfilUsuario['NumeroDePortas'] = int(input('Numero de Portas: '))
perfilUsuario['Finalidade'] = int(input('Finalidade: '))
perfilUsuario['Combustivel'] = int(input('Combustivel: '))
perfilUsuario['Valor'] = int(input('Valor: '))

user = []
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

    with open('C:/Users/User Acer/Documents/GitHub/CarRecommenderML/Datasets/NaiveBayes/DatasetCarrosComClasse.csv', 'a') as f:
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

    with open('C:/Users/User Acer/Documents/GitHub/CarRecommenderML/Datasets/NaiveBayes/DatasetCarrosComClasse.csv', 'a') as f:
        writer = csv.writer(f)
        for i in range(len(linhasinvalidas)):
            writer.writerow(linhasinvalidas[i])



# Machine Learning

dataset = pd.read_csv("C:/Users/User Acer/Documents/GitHub/CarRecommenderML/Datasets/NaiveBayes/DatasetCarrosComClasse.csv")
df = pd.DataFrame(dataset, columns=['Marca','Modelo','Ano','Concessionaria','Conforto','Seguranca','GastosFixos','Desempenho','Carroceria','NumLugares','NumeroDePortas','Finalidade','Combustivel','Valor','Valido'])



#                               Divide o dataset em treino e teste

train, test = train_test_split(df, test_size = 0.33, random_state = 42)

#                               Divide o dataset em treino e teste

TargetIndex = train[['Valido']]
TestIndex = test[['Valido']]

target_train= list(TargetIndex.index.values)
features_train= train[['Marca','Modelo','Ano','Conforto','Seguranca','GastosFixos','Desempenho','Carroceria','NumLugares','NumeroDePortas','Finalidade','Combustivel','Valor']]
target_test= list(TestIndex.index.values)
features_test= test[['Marca','Modelo','Ano','Conforto','Seguranca','GastosFixos','Desempenho','Carroceria','NumLugares','NumeroDePortas','Finalidade','Combustivel','Valor']]
'''
print(target_train)
print("funciona ate aqui")
print(features_train)
print("funciona ate aqui")
print(target_test)
print("funciona ate aqui")
print(features_test)
'''
#                               Construindo e treinando o Modelo
'''
bayes = GaussianNB()
model = bayes.fit(features_train,target_train)

#                               Fazer previsoes

preds = bayes.predict(features_test)
print(preds)

#                                   Printa as Previsoes

for car in preds:
    print(car, list(recommend.iloc[car]))





#                                   Recomenda

perfilUsuario = {"Carroceria": 1.5,"NumLugares": 5,"NumeroDePortas": 3,"Finalidade": 1,"Combustivel": 2,"Valor": 4,"Conforto": 2,"Seguranca": 2,"GastosFixos": 1,"Desempenho": 1}

usuario = [list(perfilUsuario.values())]

resultado = model.predict(usuario)

print(list(recommend.iloc[resultado[0]]))

#print(list(recommend.iloc[resultado]))
'''
#                               Main
#print(train)
adicionaItens()
print("ta rodando")