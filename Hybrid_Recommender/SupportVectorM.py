'''
References:
http://scikit-learn.org/stable/auto_examples/svm/plot_svm_kernels.html
'''
# For mathematical calculation
import numpy as np

# For plotting graphs
import matplotlib.pyplot as plt

# Import the sklearn.SVM for SVC
from sklearn.svm import SVC

# For creating datasets
from sklearn.datasets import make_circles

#
from pandas import Series, DataFrame
import pandas as pd
import csv
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
# Create the dataset




datasetML = pd.read_csv("C:/Users/User Acer/Documents/GitHub/CarRecommenderML/Datasets/SVM/DatasetCarrosComClasse.csv")
df = pd.DataFrame(datasetML, columns=['Marca','Modelo','Ano','Conforto','Seguranca','GastosFixos','Desempenho','Carroceria','NumLugares','NumeroDePortas','Finalidade','Combustivel','Valor','Valido'])

toSVM = df[['Conforto','Seguranca','GastosFixos','Desempenho','Carroceria','NumLugares','NumeroDePortas','Finalidade','Combustivel','Valor','Valido']]
'''
df['Conforto'] = df[['SensorDeEstacionamento','ArCondicionado','Bluetooth','Direcao','BancoCouro','GPS','VidrosEletricos','PilotoAutomatico','TetoSolar','TamPortaMala','Cacamba','ComputadorBordo']].mean(axis=1)
df['Seguranca'] = df[['DisponibiliPeca','Airbag','ABS','Blindagem','FarolNeblina']].mean(axis=1)
df['GastosFixos'] = df[['IPVA','ConsumoGasolina','Seguro','Manutencao']].mean(axis=1)
df['Desempenho'] = df[['IPVA','ConsumoGasolina','Seguro','Manutencao']].mean(axis=1)

df.drop(['SensorDeEstacionamento','ArCondicionado','Bluetooth','Direcao','BancoCouro','GPS','VidrosEletricos','PilotoAutomatico','TetoSolar','TamPortaMala','Cacamba','ComputadorBordo','DisponibiliPeca','Airbag','ABS','Blindagem','FarolNeblina','IPVA','ConsumoGasolina','Seguro','Manutencao','Motor','CavalosForca','VelMaxima'], axis=1, inplace=True)
'''
carrosAExibir = df[['Marca','Modelo','Ano']]
#Recomendacoes e Novo Dataset
#                               Faz Recomendacoes para alimentar um dataset que sera utilizado no modelo de machine learning
key = ["Conforto","Seguranca","GastosFixos","Desempenho","Carroceria","NumLugares","NumeroDePortas","Finalidade","Combustivel","Valor"]
#perfilUsuario = {"Conforto": 2,"Seguranca": 2,"GastosFixos": 1,"Desempenho": 1,"Carroceria": 1.5,"NumLugares": 5,"NumeroDePortas": 3,"Finalidade": 1,"Combustivel": 2,"Valor": 4}
perfilUsuario = {}
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

    user = []
    for j in key: #O perfil do usuario recebe os valores inseridos
        user.append(perfilUsuario[j])

X = toSVM.drop('Valido', axis=1)
Y = toSVM['Valido']
kernals = ['linear','poly','rbf', 'sigmoid', ]

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.20)

for kernal in kernals:

    print("Using %s kernel" %kernal)

    svclassifier = SVC(kernel=kernal)
    svclassifier.fit(X_train, y_train)

    y_pred = svclassifier.predict(X_test)

    print(confusion_matrix(y_test,y_pred))
    print(classification_report(y_test,y_pred))

    print(y_pred)


























'''
# Plot the dataset
plt.scatter(df[:,0],df[:,1],c=value)
plt.show()

# Calculate the higher dimension value
x = df[:,0]
y = df[:,1]
z = x**2 + y**2

kernals = ['linear','poly','rbf']
training_set = np.c_[x,y]

# Train and predict for each kernal
for kernal in kernals:
    clf=svm.SVC(kernel=kernal,gamma=2)

    # Train the model
    clf.fit(training_set,value)

    # Test the model
    prediction = clf.predict([[-0.4,-0.4]])

    print(prediction)

    Output:
    [0] - linear kernal
    [1] - polynomial kernal
    [1] - rbf kernal



    # plot the line, the points, and the nearest vectors to the plane
    X = training_set
    y = value
    X0 = X[np.where(y == 0)]
    X1 = X[np.where(y == 1)]
    plt.figure()
    x_min = X[:, 0].min()
    x_max = X[:, 0].max()
    y_min = X[:, 1].min()
    y_max = X[:, 1].max()
    XX, YY = np.mgrid[x_min:x_max:200j, y_min:y_max:200j]
    Z = clf.decision_function(np.c_[XX.ravel(), YY.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(XX.shape)
    plt.pcolormesh(XX, YY, Z > 0, cmap=plt.cm.Paired)
    plt.contour(XX, YY, Z, colors=['k', 'k', 'k'], linestyles=['--', '-', '--'],
                levels=[-.5, 0, .5])

    plt.scatter(X0[:, 0], X0[:, 1], c='r',s=50)
    plt.scatter(X1[:, 0], X1[:, 1], c='b',s=50)
    title = ('SVC with {} kernal').format(kernal)
    plt.title(title)
    plt.show()

'''