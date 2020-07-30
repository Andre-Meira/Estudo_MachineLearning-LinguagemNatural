# C:\Users\andre\PycharmProjects\programas\IA

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import nltk
# from nltk import ngrams
from string import punctuation


resenha = pd.read_csv("imdb-reviews-pt-br.csv")

classficacao = resenha["sentiment"].replace(["neg", "pos"], [0, 1])

resenha["classficacao"] = classficacao

texto = ["Assisti um filme ótimo", "Assisti um filme ruim"]


def classificar_text(text, coluna_texto, coluna_classificacao):
    vetorizar = CountVectorizer(lowercase=False,
                                max_features=50)

    bag_of_words = vetorizar.fit_transform(resenha[coluna_texto])
    train, test, class_train, class_test = train_test_split(bag_of_words,
                                                            text[coluna_classificacao],
                                                            random_state=50)
    regresssao_logisct = LogisticRegression(solver='lbfgs')
    regresssao_logisct.fit(train, class_train)
    return regresssao_logisct.score(test, class_test)


def nuvem_palavras_neg(text, coluna_texto):
    # SEPARA OS TEXO NEGATIVO DOS POSITIVOS(ESSE É O NEGATIVO)

    texto_negativo = text.query("sentiment=='neg'")

    todas_palavras = ' '.join([text for txt in texto_negativo[coluna_texto]])

    nuvem_palavras = WordCloud(width=800,
                               height=500,
                               max_font_size=110,
                               collocations=False).generate(todas_palavras)

    plt.figure(figsize=(1, 7))  # cria a imagem
    plt.imshow(nuvem_palavras, interpolation='bilinear')  # qual conteudo vai se transformado em imagem
    plt.show()  # mostra a imagem


def nuvem_palavras_pos(text, coluna_texto):
    # SEPARA OS TEXO NEGATIVO DOS POSITIVOS(ESSE É O POSITIVO)

    texto_positivo = text.query("sentiment == 'pos'")

    todas_palavras = ''.join([text for txt in texto_positivo[coluna_texto]])

    nuvem_palavras = WordCloud(width=800,
                               height=500,
                               max_font_size=110,
                               collocations=False).generate(todas_palavras)
    plt.figure(figsize=(1, 7))  # cria a imagem
    plt.imshow(nuvem_palavras, interpolation='bilinear')  # qual conteudo vai se transformado em imagem
    plt.show()  # mostra a imagem


def frenq_palavra(frase_f):
    # VE A FREQUENCIA EM QUE AS PALAVRAS APARENCEM
    frequencia = nltk.FreqDist(frase_f)
    return frequencia


def separador_frase(frase):
    # SEPARA FRASE EM PALAVRAS
    token_space = nltk.tokenize.WordPunctTokenizer()
    token_frase = token_space.tokenize(frase)
    return token_frase


def remove_palavras_irrelevante():

    palavras_irrelevantes = nltk.corpus.stopwords.words("portuguese")
    frase_processada = list()

    for opiniao in resenha.text_pt:

        nova_frase = list()
        opiniao = opiniao.lower()
        palavras_texto = separador_frase(opiniao)

        for palavra in palavras_texto:

            if palavra not in palavras_irrelevantes:
                nova_frase.append(palavra)

        frase_processada.append(' '.join(nova_frase))

    resenha["Tratamento"] = frase_processada

    # ----------DAQUI PRA CIMA REMOVE PALAVRAS IRRELAVANTES

    # ----------DAQUI PRA BAIXO REMOVE PONTUAÇAO=

    pontuacao = list()

    for ponto in punctuation:
        pontuacao.append(ponto)

    pontuacao_stopwords = pontuacao + palavras_irrelevantes

    frase_processada = list()

    for opiniao in resenha["Tratamento"]:

        nova_frase = list()
        palavras_texto = separador_frase(opiniao)

        for palavra in palavras_texto:

            if palavra not in pontuacao_stopwords:
                nova_frase.append(palavra)

        frase_processada.append(' '.join(nova_frase))

    resenha["Tratamento_2"] = frase_processada
    # --- RETIRA AS PALAVRAS COM SEMANTICA PARECIDA

    stemmer = nltk.RSLPStemmer()

    frase_processada = list()

    for opiniao in resenha["Tratamento_2"]:

        nova_frase = list()
        palavras_texto = separador_frase(opiniao)

        for palavra in palavras_texto:

            if palavra not in pontuacao_stopwords:
                nova_frase.append(stemmer.stem(palavra))

        frase_processada.append(' '.join(nova_frase))

    resenha["Tratamento_3"] = frase_processada


def ngramis(text, colouna_text, coluna_classficacao):
    tfidf = TfidfVectorizer(lowercase=False, ngram_range=(1, 2))
    vetor_tfidf = tfidf.fit_transform(text[colouna_text])
    treino, test, class_treino, class_test = train_test_split(vetor_tfidf,
                                                              text[coluna_classficacao],
                                                              random_state=42)

    regressao_logistica = LogisticRegression()
    regressao_logistica.fit(treino, class_treino)
    acuracia_tfidf = regressao_logistica.score(test, class_test)
    # previw = regressao_logistica.predict(test)
    return acuracia_tfidf


# print(ngramis(resenha, "text_pt", "classficacao"))

# test4 = nltk.corpus.stopwords.words("portuguese")
