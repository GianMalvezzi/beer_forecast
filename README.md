# A ML model API for beer consumption forecasting

[PT-BR]
Este projeto tem o intuito de criar uma API de ML para prever o consumo de cerveja de uma região de São Paulo, Brasil. Usando a linguagem python e diversas bibliotecas como scikit-learn para a criação do modelo e Flask para a criação da API, é feito uma previsão baseada em três modelos: Regressão de Lasso, Regressão usando Random Forest (Random Forest Regressor) e Regressão usando uma Support Vector Machine (Support Vector Regressor). A escolha de uma regressão linear ao invés de uma previsão de série temporal foi adotada baseada nos dados retirados no arquivo research.

Dados para o treinamento do [modelo](https://www.kaggle.com/datasets/dongeorge/beer-consumption-sao-paulo) e para a criação do dataset de [feriados](https://www.kaggle.com/code/mpwolke/s-o-paulo-city-holidays/notebook) foram retirados do Kaggle

[EN-US]
This project aims to create an ML API to predict beer consumption in a region of São Paulo, Brazil. Using the python language and several libraries like scikit-learn for the model creation and Flask for the API creation, a prediction is made based on three models: Lasso Regression, Regression using Random Forest (Random Forest Regressor) and Regression using a Support Vector Machine (Support Vector Regressor). The choice of a linear regression instead of a time series forecast was adopted based on the data drawn from the research file.

Data for training the [model](https://www.kaggle.com/datasets/dongeorge/beer-consumption-sao-paulo) and for creating the [holiday](https://www.kaggle.com/code/mpwolke/s-o-paulo-city-holidays/notebook) dataset were taken from Kaggle