import pandas as pd
import numpy as np
import json
import re
import os, sys
from time import time
#ML
import multiprocessing
from gensim.models import Word2Vec
import nltk
from nltk.tokenize import word_tokenize
#Plot
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import streamlit as st



#Dictionnaires
#Flamanville
with open('/home/herrem/Documents/ASN/ASN_LDA_dict_utf8') as f:
   data = json.load(f)
df1 = pd.DataFrame(data)
df1.columns = ['Référence', 'Titre', """Date d'inspection""", 'Lien url', 'Contenu']
data_flam = df1['Contenu'].values.tolist()


def pause():
    Pause = input("Appuyer sur entrée pour continuer...")
    return Pause

def phrases(doc):
    sentences = nltk.tokenize.sent_tokenize(doc)
    return sentences

def extract_demandes(doc):
    demand =[]
    for element in doc:
        if re.findall(r'([^.]*je vous demande[^.]*)', element):
            tokens = word_tokenize(element)
            tokens = [x.lower() for x in tokens]
            tokens = [item for item in tokens if item.isalpha() and len(item)>2]
            #print(element)
            demand.append(tokens)
        else:
            continue
    return demand

def tsne_plot(model,perplex):
    "Creates and TSNE model and plots it"
    labels = []
    tokens = []

    for word in model.wv.vocab:
        tokens.append(model[word])
        labels.append(word)

    tsne_model = TSNE(perplexity=perplex, n_components=2, init='pca', n_iter=5000)
    new_values = tsne_model.fit_transform(tokens)

    x = []
    y = []
    for value in new_values:
        x.append(value[0])
        y.append(value[1])

    plt.figure(figsize=(16, 16))
    for i in range(len(x)):
        plt.scatter(x[i], y[i])
        plt.annotate(labels[i],
                     xy=(x[i], y[i]),
                     xytext=(5, 2),
                     textcoords='offset points',
                     ha='right',
                     va='bottom')

    plt.savefig('TSNE_w2v_2_p={}.png'.format(perplex))

t1 =time()


demandes = []
words = []
for item in data_flam:
    sentences = phrases(item)
    demandes.append(extract_demandes(sentences))
    for element in sentences:
        #demandes.append(extract_demandes(element))
        tokens = word_tokenize(element)
        tokens = [x.lower() for x in tokens]
        tokens = [item for item in tokens if item.isalpha() and len(item) > 3]
        words.append(tokens)


print("-------------------")
print(np.size(words))




patterns = [' délit ',' mise en demeure ',' l.596-4 ',' sanctions ',' sanction ',' infraction ',' infractions ',' pénal ',' pénale ',' pénales ',' amende '
           ]
result=[]
for element in data_flam:
    for pattern in patterns:
        if re.findall(pattern, element):
            sentences = phrases(element)
            result.append(element)
            for item in sentences:
                if re.findall(pattern, item):
                    print(' ')

for i in range(0, len(result),1):
    print(result[i])
    print("--------------------")

pause()
print(np.size(demandes))
print('Time to preprocess the data: {} mins'.format(round((time() - t1)/60, 2)))

cores = multiprocessing.cpu_count()
w2v_model = Word2Vec(min_count=1,
                     window=6,
                     size=300,
                     sample=6e-5,
                     alpha=0.03,
                     min_alpha=0.0007,
                     negative=20,
                     workers=cores-1)

w2v_model.build_vocab(words, progress_per=10000)

t2 = time()
w2v_model.train(words, total_examples=w2v_model.corpus_count, epochs=50, report_delay=1)
print('Time to train the model: {} mins'.format(round((time() - t2)/60, 2)))
#print(w2v_model.wv.vocab)

w2v_model.init_sims(replace=True)

w2v_model.save("word2vec.model")

for i in range(1,20,1):
    tsne_plot(w2v_model,i)

i=0
for i in range(0,1000,1):
    text=input('Saisir un mot :\n')
    print(w2v_model.wv.most_similar(positive=[text],topn=20))
    print(type(w2v_model.wv.most_similar(positive=[text],topn=20)))
    i+=1

print('vector size :', w2v_model.wv.vector_size)
