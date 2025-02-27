English description below:

Classificação de Câncer de Mama com Redes Neurais

Sobre o Projeto

Neste projeto, desenvolvi um modelo de aprendizado de máquina 
utilizando redes neurais para classificar tumores de mama como malignos ou benignos com base no conjunto de dados Breast Cancer Wisconsin do sklearn.datasets.

O que eu fiz


Carregamento dos Dados

Utilizei o conjunto de dados load_breast_cancer() da biblioteca sklearn.

Extraí as features (X), os nomes das classes (names) e os rótulos (y).


Pré-processamento

Normalizei os dados usando StandardScaler para garantir que todas as features estivessem na mesma escala.

Dividi os dados em treino e teste (train_test_split) para avaliar o modelo.

Converti os rótulos para one-hot encoding com to_categorical().

Construção do Modelo de Rede Neural

Criei um modelo Sequential do Keras, composto por múltiplas camadas densas (Dense) e camadas de Dropout para evitar overfitting.

Usei a função de ativação ReLU nas camadas ocultas e sigmoid na saída para classificação binária.

Treinamento do Modelo

Compilei o modelo com o otimizador Nadam e a função de perda binary_crossentropy.

Treinei por 20 épocas, salvando o melhor modelo usando ModelCheckpoint.

Avaliação e Predição

Avaliei o modelo no conjunto de teste (model.evaluate()), atingindo uma acurácia superior a 99%.

Criei uma função prediction(dados) para fazer previsões individuais, fornecendo as probabilidades de malignidade e benignidade.


----------------------------------------------------------------------------------------------

Breast Cancer Classification with Neural Networks

About the Project

In this project, I developed a machine learning model using neural networks to classify breast tumors as malignant or benign based on the Breast Cancer Wisconsin dataset from sklearn.datasets.

What I Did

Data Loading

I used the load_breast_cancer() dataset from the sklearn library.

I extracted the features (X), class names (names), and labels (y).

Preprocessing

I normalized the data using StandardScaler to ensure that all features were on the same scale.

I split the data into training and testing sets (train_test_split) to evaluate the model.

I converted the labels to one-hot encoding using to_categorical().

Neural Network Model Construction

I created a Sequential model in Keras, consisting of multiple dense layers (Dense) and dropout layers to prevent overfitting.

I used the ReLU activation function in the hidden layers and sigmoid in the output for binary classification.

Model Training

I compiled the model with the Nadam optimizer and the binary_crossentropy loss function.

I trained the model for 20 epochs, saving the best model using ModelCheckpoint.

Evaluation and Prediction

I evaluated the model on the test set (model.evaluate()), achieving an accuracy of over 99%.

I created a function prediction(data) to make individual predictions, providing the probabilities of malignancy and benignity.


