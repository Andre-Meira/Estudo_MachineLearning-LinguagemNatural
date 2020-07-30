from sklearn.svm import LinearSVC  # MODULO DE APRENDIZADO LINEAR
from sklearn.metrics import accuracy_score  # TRAZ O ACERTO DA MAQUINA
from sklearn.model_selection import train_test_split  # SEPARA TRAIN DE TEST
from sklearn.svm import SVC  # MODULO DE APRENDIZADO MAIS CORRETO
from sklearn.tree import DecisionTreeClassifier  # ARVORE DE DESCISAO
import pandas as pd  # PANDAS
import seaborn as sns
import numpy as np  # NUMPY
from datetime import datetime  # PEGA A HORA/DIA/ANO ATUAL
from sklearn.preprocessing import StandardScaler  # FAZ O ESCALAMENTO
# from sklearn.cluster import KMeans  # SEPARA EM GRUPOS
# from sklearn.cluster import AgglomerativeClustering  # modeulo de hierarquia


class Ur1:
    var = ''

    def estudos1(self):
        self.var = 0

        uri = "https://gist.githubusercontent.com/guilhermesilveira/2d2efa37d66b6c84a722ea627a897ced/raw/" \
              "10968b997d885cbded1c92938c7a9912ba41c615/tracking.csv"

        dados = pd.read_csv(uri)  # IMPORTANDO A URI

        mapa = {
            "home": "Principal",
            "how_it_works": "Como_Funciona",
            "contact": "Contado",
            "bought": "Comprou"
        }  # MAPA COM O NOVOS NOMES PARA AS COLUNAS

        dados = dados.rename(columns=mapa)  # ALTERANDO O NOME DA COLUNA

        x = dados[["Principal", "Como_Funciona", "Contado"]]  # SEPARANDO AS COLUNAS
        y = dados["Comprou"]  # SEPANDO AS COLUNAS

        print(dados.head())

        # (treino_x = x[:75] #75% PARA TREINO
        # treino_y = y[:75] #75% PARA TREINO

        # test_x = x[75:] #25% PARA TESTE
        # test_y = y[75:] #25% PARA TESTE)

        treino_x, test_x, treino_y, test_y = train_test_split(
            x, y, random_state=42,
            test_size=0.25,
            stratify=y)

        # FUNÇÃO PARA SEPARAR TREINO DE TEST(IGUAL O CODIGO ACIMA) SO QUE ALEATORIAMENTE
        # SE EU PASSO UMA SEED ELE TIRA ALEATORIADADE
        # STRATIFY E PROPORÇÃO QUE ELE SEPARA

        modelo = LinearSVC()
        modelo.fit(treino_x, treino_y)  # TREINADO EM CIMA DO TREINO_X e TREINO_Y

        previsoes = modelo.predict(test_x)  # RETORNA O RESUTADO DA IA

        taxa_de_acerto = accuracy_score(test_y, previsoes)  # RETORNA A TAXA DE ACERTO DA IA

        print(taxa_de_acerto * 100)


class ur2:
    var = ''

    def estudos2(self):
        self.var = 0

        uri = "https://gist.githubusercontent.com/guilhermesilveira/1b7d5475863c15f484ac495bd7097" \
              "5cf/raw/16aff7a0aee67e7c100a2a48b676a2d2d142f646/projects.csv"

        dados = pd.read_csv(uri)

        a_renomear = {
            "unfinished": "Nao_finalizado",
            "price": "Preco",
            "expected_hours": "Horas_esperadas"

        }

        dados = dados.rename(columns=a_renomear)

        modelo = LinearSVC()

        troca = {
            0: 1,
            1: 0
        }  # COMPLEMENTAÇÃO DO NAO FINALIZAR

        dados['Finalizados'] = dados.Nao_finalizado.map(troca)
        # ADICONA UMA NOVA COLUNA PARA COMPLEMENTAR O "Nao_finalizado"

        sns.scatterplot(x='Horas_esperadas', y='Preco', data=dados)
        sns.relplot(x='Horas_esperadas', y='Preco', hue='Finalizados', col='Finalizados', data=dados)

        x = dados[['Horas_esperadas', 'Preco']]
        y = dados['Finalizados']

        treino_x, test_x, treino_y, test_y = train_test_split(
            x, y,
            test_size=0.25,
            stratify=y)

        modelo.fit(treino_x, treino_y)
        previsoes = modelo.predict(test_x)

        acuracia = accuracy_score(test_y, previsoes)

        print(acuracia)


class Ur3:
    var = 0

    def ia_svc(self):
        self.var = 0

        uri = 'https://gist.githubusercontent.com/guilhermesilveira/4d1d4a16ccbf6ea4e0a64a38a24ec884/' \
              'raw/afd05cb0c796d18f3f5a6537053ded308ba94bf7/car-prices.csv'

        dados = pd.read_csv(uri)

        trocar = {
            'mileage_per_year': 'Milhas_por_anos',
            'model_year': 'Modelo_veiculo',
            'price': 'Preco',
            'sold': 'Vendido'
        }

        dados = dados.rename(columns=trocar)

        a_trocar = {
            'no': 0,
            'yes': 1
        }

        dados.Vendido = dados.Vendido.map(a_trocar)

        ano_atual = datetime.today().year
        dados['idade_V'] = ano_atual - dados.Modelo_veiculo
        dados['km_ano'] = dados.Milhas_por_anos * 1.60934

        dados = dados.drop(columns=['Unnamed: 0', 'Milhas_por_anos', 'Modelo_veiculo'], axis=1)

        print(dados.head())

        x = dados[["Preco", "idade_V", "km_ano"]]
        y = dados["Vendido"]

        seed = 5
        np.random.seed(seed)

        raw_treino_x, raw_test_x, treino_y, test_y = train_test_split(
            x, y,
            train_size=0.25,
            stratify=y
        )

        scaler = StandardScaler()
        scaler.fit(raw_treino_x)
        treino_x = scaler.transform(raw_treino_x)
        test_x = scaler.transform(raw_test_x)

        modelo = SVC()
        modelo.fit(treino_x, treino_y)
        prediction = modelo.predict(test_x)

        print(accuracy_score(test_y, prediction) * 100)

    def decision_tree(self):
        self.var = 0

        uri = 'https://gist.githubusercontent.com/guilhermesilveira/4d1d4a16ccbf6e' \
              'a4e0a64a38a24ec884/raw/afd05cb0c796d18f3f5a6537053ded308ba94bf7/car-prices.csv'
        dados = pd.read_csv(uri)

        trocar = {
            'mileage_per_year': 'Milhas_por_anos',
            'model_year': 'Modelo_veiculo',
            'price': 'Preco',
            'sold': 'Vendido'
        }

        dados = dados.rename(columns=trocar)

        a_trocar = {
            'no': 0,
            'yes': 1
        }

        dados.Vendido = dados.Vendido.map(a_trocar)

        ano_atual = datetime.today().year
        dados['idade_V'] = ano_atual - dados.Modelo_veiculo
        dados['km_ano'] = dados.Milhas_por_anos * 1.60934

        dados = dados.drop(columns=['Unnamed: 0', 'Milhas_por_anos', 'Modelo_veiculo'], axis=1)

        print(dados.head())

        x = dados[["Preco", "idade_V", "km_ano"]]
        y = dados["Vendido"]

        seed = 5
        np.random.seed(seed)

        raw_treino_x, raw_test_x, treino_y, test_y = train_test_split(
            x, y,
            train_size=0.25,
            stratify=y
        )

        scaler = StandardScaler()  # modulo de escalamento
        scaler.fit(raw_treino_x)
        treino_x = scaler.transform(raw_treino_x)
        test_x = scaler.transform(raw_test_x)

        modelo = DecisionTreeClassifier()
        modelo.fit(treino_x, treino_y)
        prediction = modelo.predict(test_x)

        print(accuracy_score(test_y, prediction) * 100)


class NaoSuperV:
    uri = 'https://raw.githubusercontent.com/alura-cursos/machine-learnin' \
          'g-algoritmos-nao-supervisionados/master/movies.csv'

    dados = pd.read_csv(uri)

    a_trocar = {
        "title": "Titulo",
        "movieId": "FilmeID",
        "genres": "Genero"
    }

    dados = dados.rename(columns=a_trocar)
    generos = dados.Genero.str.get_dummies()  # extrai as cotergorias em varias
    print(generos)
