#importando as bibliotecas
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

#importando o modelo Naive Bayes
from sklearn.naive_bayes import GaussianNB


#carregando os dados
X, y = load_iris(return_X_y=True)

#separando a base de treino da validação
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0)

#criando o classificador
gnb = GaussianNB()

#treinando o classificador
gnb.fit(X_train, y_train)

#realizando as previsões
y_pred = gnb.predict(X_test)