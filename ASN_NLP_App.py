#Frameworks
import streamlit as st
import streamlit.components.v1 as components
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import json
import os
import subprocess
import re
import sys
from PIL import Image
import webbrowser
import nltk
import pandas as pd
from joblib import load
import pickle
from sklearn.metrics.pairwise import euclidean_distances, cosine_distances

os.getcwd()
reload(sys)
sys.setdefaultencoding('utf8')

#------------Chargemenent des données----------------#
#Images
#Images d'illustration
Logo_ASN = Image.open('/home/herrem/Images/Logo_ASN.png')
image3 = Image.open('/home/herrem/Images/LDA_principe.jpg')

#Images TSNE évolution
TSNE_5 = Image.open('/home/herrem/Images/TSNE ASN GIF/perplexity_5.png')
TSNE_10 = Image.open('/home/herrem/Images/TSNE ASN GIF/perplexity_10.png')
TSNE_15 = Image.open('/home/herrem/Images/TSNE ASN GIF/perplexity_15.png')
TSNE_20 = Image.open('/home/herrem/Images/TSNE ASN GIF/perplexity_20.png')
TSNE_25 = Image.open('/home/herrem/Images/TSNE ASN GIF/perplexity_25.png')
TSNE_30 = Image.open('/home/herrem/Images/TSNE ASN GIF/perplexity_30.png')
TSNE_35 = Image.open('/home/herrem/Images/TSNE ASN GIF/perplexity_35.png')
TSNE_40 = Image.open('/home/herrem/Images/TSNE ASN GIF/perplexity_40.png')
TSNE_45 = Image.open('/home/herrem/Images/TSNE ASN GIF/perplexity_45.png')
TSNE_50 = Image.open('/home/herrem/Images/TSNE ASN GIF/perplexity_50.png')

#Dictionnaires
#Flamanville
with open('/home/herrem/Documents/ASN/ASN_LDA_dict_flammanville_utf8') as f:
   data = json.load(f)
df1 = pd.DataFrame(data)
df1.columns = ['Référence', 'Titre', """Date d'inspection""", 'Lien url', 'Contenu']
data_flam = df1['Contenu'].values.tolist()

#All
with open('/home/herrem/Documents/ASN/ASN_LDA_dict_utf8') as f:
   data_all = json.load(f)
df2 = pd.DataFrame(data_all)
df2.columns = ['Référence', 'Titre', """Date d'inspection""", 'Lien url', 'Contenu']

#LDA
#Load model
model = load('/home/herrem/Documents/ASN/LDA_flam.pickle')
print(model.get_params())
vectorizer = pickle.load(open("/home/herrem/Documents/ASN/vectorizer_flam.pickle","rb"))
#Handle data
data_vectorized = vectorizer.transform(data_flam)
lda_output = model.transform(data_vectorized)
#------------------Fonctions-----------------#
# Show top n keywords for each topic
def show_topics(vectorizer, lda_model, n_words):
    keywords = np.array(vectorizer.get_feature_names())
    topic_keywords = []
    for topic_weights in lda_model.components_:
        top_keyword_locs = (-topic_weights).argsort()[:n_words]
        topic_keywords.append(keywords.take(top_keyword_locs))
    return topic_keywords

#L2 Distance
def general_euclidian_dist(index,doc_matrix,n_top):
    dists = euclidean_distances(doc_matrix[index,:].reshape(1, -1), doc_matrix)[0]
    doc_ids = np.argsort(dists)[:n_top]
    return doc_ids

#Cosine distance
def general_cosine_dist(index,doc_matrix,n_top):
    dists = cosine_distances(doc_matrix[index,:].reshape(1, -1), doc_matrix)[0]
    doc_ids = np.argsort(dists)[:n_top]
    return doc_ids

#Pre-process
topic_keywords = show_topics(vectorizer=vectorizer, lda_model=model, n_words=15)

# Topic - Keywords Dataframe
df_topic_keywords = pd.DataFrame(topic_keywords)
df_topic_keywords.columns = ['Word ' + str(i) for i in range(df_topic_keywords.shape[1])]
df_topic_keywords.index = ['Topic ' + str(i) for i in range(df_topic_keywords.shape[0])]
print(df_topic_keywords)

categories = ['Topic 1', 'Topic 2', 'Topic 3', 'Topic 4', 'Topic 5', 'Topic 6', 'Topic 7', 'Topic 8', 'Topic 9','Topic 10', 'Topic 11', 'Topic 12', 'Topic 13']
#--------------Application Streamlit-------------#
#Sidebar
st.sidebar.title("Outil d'analyse des lettres d'inspection de l'ASN")
page = st.sidebar.selectbox("Menu", ["Accueil","NLP", "Exploration Simple","Exploration Avancée", "Analyse d'une lettre", "Suggestions nouveau document"])
st.sidebar.text('Avril 2020 | Rémi Martinie')
st.sidebar.image(Logo_ASN, use_column_width=True)

if page == "Accueil":
    st.title("Accueil")
    st.header("Introduction")
    st.write("Cet outil analyse les lettres d'inspection de l'Autorité de Sûreté Nucléaire (ASN) à l'aide de techniques de Natural Language Processing (NLP). Les missions de l'ASN s'articulent autour de trois métiers « historiques » : ")
    st.write("La réglementation : l'ASN est chargée de contribuer à l'élaboration de la réglementation, en donnant son avis au Gouvernement sur les projets de décrets et d'arrêtés ministériels ou en prenant des décisions réglementaires à caractère technique")
    st.write("Le contrôle : l'ASN est chargée de vérifier le respect des règles et des prescriptions auxquelles sont soumises les installations ou activités qu'elle contrôle.")
    st.write("L'information du public : l'ASN est chargée de participer à l'information du public, y compris en cas de situation d'urgence.")


if page == "NLP":
    st.title("Natural Language Processing")
    st.write("""Le Natural Language Processing (NLP) autrement appelé en français “Traitement automatique du langage naturel” est une branche très importante du Machine Learning et donc de l’intelligence artificielle. Le NLP est la capacité d’un programme à comprendre le langage humain.""")
    st.write("""Prenons quelques exemples pratiques qu’on utilise tous les jours pour mieux comprendre :""")
    st.write("""Les spams : toutes les boîtes mails utilisent un filtre antispam et cela fonctionne avec le filtrage bayésien en référence au théorème de Bayes qui est une technique statistique de détection de spams. Ces filtres vont “comprendre” le texte et trouver s’il y a des corrélations de mots qui indiquent un pourriel.""")
    st.write("""Google Traduction : vous avez probablement tous utilisé ce système et leur technologie utilise de nombreux algorithmes dont du NLP. Ici, le défi n’est pas de traduire le mot, mais de garder le sens d’une phrase dans une autre langue.""")
    st.write("""Le logiciel Siri créé par Apple ou Google Assistant utilise du NLP pour traduire du texte transcrit en du texte analysé afin de vous donner une réponse adaptée à votre demande.""")
    st.title("Allocation de Dirichlet Latente")
    st.image(image3, use_column_width=True)
    st.header("Optimisation des modèles")
    st.write("Un modèle de LDA s'optimise au travers de deux paramètres : la perplexité et l'intuition humaine")
    st.subheader("Perplexité")
    st.write("La perplexité est une fonction décroissante de la log-vraisemblance L(w) des documents invisibles wd. Plus la perplexité est faible, meilleur est le modèle.")
    st.latex(r'''
             Perplexity (\textbf{w}) = e^{-\frac{\mathscr{L}(\textbf{w})}{count of tokens}}
             ''')
    X = [.3,.4,.5,.6,.7,.8,.9]
    Y = [5,10,15,20,25,30,35,40,45,50,55,60]
    Z = [[1353,1271,1181,1154,1144,1145,1152],
         [1289,1190,1120,1094,1087,1089,1098],
         [1273,1167,1097,1069,1061,1061,1066],
         [1256,1144,1077,1052,1043,1046,1051],
         [1251,1146,1072,1044,1033,1034,1043],
         [1243,1135,1050,1029,1023,1021,1034],
         [1415,1191,1088,1049,1030,1030,1037],
         [2012,1434,1230,1153,1118,1022,1048],
         [2167,1500,1257,1169,1134,1019,1038],
         [2340,1559,1291,1195,1159,1034,1037],
         [2555,1655,1336,1215,1173,1033,1037],
         [2797,1734,1354,1235,1182,1037,1033]]

    learning_decay= st.slider('Learning decay', min(X), max(X),(X[0], X[-1]), step=.1)
    n_topics = st.slider('Nombre de topics', min(Y), max(Y), (Y[0], Y[-1]), step=5)
    st.write(n_topics[0])
    st.write(n_topics[1])
    c=0
    for element in Y:
        if element == n_topics[1]:
            low_bound_top = c
        if element == learning_decay[1]:
            low_bound_lear = c
        c+=1
    layout = go.Layout(scene=dict(
            xaxis=dict(title='Learning decay'),
            yaxis=dict(title='Nombre de topics'),
            zaxis=dict(title='Perplexity'),
        ),
    )
    fig = go.Figure(data=[go.Surface(x=X, y=Y, z=Z)],layout=layout)
    fig.update_layout(title='', autosize=False,
                    width=750, height=750,
                    template="plotly_white",
                    margin=dict(l=65, r=50, b=65, t=90))

    st.plotly_chart(fig)
    st.subheader("Intuition humaine")
    st.write("Après l'optimisation des paramètres statistiques, seule l'intuition humaine est capable d'estimer la qualité des thèmes obtenus")
    if st.button('Afficher une visualisation de LDA'):
        webbrowser.open_new_tab(r'/home/herrem/Documents/ASN/LDA_v2_topic_35_maxdf05_mindf40_bigrams.html')
    if st.button('Afficher un TSNE'):
        webbrowser.open_new_tab(r'/home/herrem/Documents/ASN/TSNE_ASN_topics35_perplexity30_maxdf02.html')
    perplexity_range = st.slider('Perplexity TSNE', 5, 50, 5, step=5)
    st.image(Image.open('/home/herrem/Images/TSNE ASN GIF/perplexity_{}.png'.format(perplexity_range)), use_column_width=True)

if page == "Exploration Simple":
    st.header("Exploration simple")
    search = st.text_input("""Moteur de recherche sur le contenu d'une lettre""", value='', key=None, type='default')
    INB = st.radio("Sélectionner l'INB",('Flamanville','Tout'))
    if INB == 'Flamanville':
        st.subheader('Résultats')
        st.write('Nombre de résultats :',df1[['Référence','Titre', """Date d'inspection""", 'Lien url']].loc[df1.Contenu.str.contains(search,case=False)].shape[0])
        st.table(df1[['Référence','Titre', """Date d'inspection""", 'Lien url']].loc[df1.Contenu.str.contains(search,case=False)])
    else:
        st.subheader('Résultats')
        st.write('Nombre de résultats :',df2[['Référence','Titre', """Date d'inspection""", 'Lien url']].loc[df2.Contenu.str.contains(search,case=False)].shape[0])
        st.table(df2[['Référence','Titre', """Date d'inspection""", 'Lien url']].loc[df2.Contenu.str.contains(search,case=False)])


if page == "Exploration Avancée":
    print(df_topic_keywords)
    st.table(df_topic_keywords)

if page == "Analyse d'une lettre":
    option = st.selectbox("Sélectionnez la lettre à analyser",df1['Référence'])
    st.write('Le document sélectionné est :', option)
    if st.button('Lire la lettre'):
        #os.startfile(r'/home/herrem/Documents/ASN/ASN_pdf/{}'.format(df1[['Référence']].loc[df1['Référence'].str.contains(option,case=False)].iloc[0][0]))
        subprocess.Popen([r'/home/herrem/Documents/ASN/ASN_pdf/{}'.format(df1[['Référence']].loc[df1['Référence'].str.contains(option,case=False)].iloc[0][0])], shell=True)
        #webbrowser.open(r'/home/herrem/Documents/ASN/ASN_pdf/{}'.format(df1[['Référence']].loc[df1['Référence'].str.contains(option,case=False)].iloc[0][0]))


    sentences = nltk.tokenize.sent_tokenize(df1[['Contenu']].loc[df1['Référence'].str.contains(option, case=False)].iloc[0][0])
    demand=[]
    for element in sentences:
        if re.findall(r'([^.]*je vous demande[^.]*)', element):
            #tokens = nltk.word_tokenize(element)
            #tokens = [x.lower() for x in tokens]
            #tokens = [item for item in tokens if item.isalpha() and len(item)>2]
            #print(element)
            demand.append(element)

    st.subheader('Synthèse des demandes')
    st.table(demand)
    tokens = nltk.tokenize.word_tokenize(df1[['Contenu']].loc[df1['Référence'].str.contains(option,case=False)].iloc[0][0])
    keywords = [item for item in tokens if item.isalpha() and len(item) > 3]
    fd = nltk.FreqDist(keywords)
    a = fd.keys()
    b = fd.values()
    words = []
    v = []
    for value in a:
        words.append(value)
    for value in b:
        v.append(value)
    fig = go.Figure()
    fig.add_trace(go.Histogram(histfunc="sum", y=v[100:], x=words[100:], name="sum"))
    st.plotly_chart(fig)
    st.subheader("Topics distribution :")
    fig2 = go.Figure(data=go.Scatterpolar(
        name="Distribution de Topics",
        r=lda_output[df1[['Référence']].index.values[df1['Référence'].str.contains(option,case=False)]][0].tolist(),
        theta=['Génie Civil', 'Radiographies', 'Effluents', 'Soudage', 'Surveillance Prestataires', 'Qualité', 'Essais', 'Structure', 'Inspection','Liner Piscine', 'Ventilation', 'Arrêté INB', 'Thermo-hydraulique'],
        fill='toself',
    ))
    fig2.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True
            ),
        angularaxis_categoryarray=['Génie Civil', 'Radiographies', 'Effluents', 'Soudage', 'Surveillance Prestataires', 'Qualité', 'Essais', 'Structure', 'Inspection','Liner Piscine', 'Ventilation', 'Arrêté INB', 'Thermo-hydraulique']
        ),
        showlegend=False,
        template = "ggplot2"
    )
    st.plotly_chart(fig2)
    st.subheader("Documents similaires :")
    dist = st.radio("Distance", ('L2', 'Cosine'))
    n_result = st.slider('Nombre de résultats à afficher :',1,50,1)
    if dist == 'L2':
        st.table(df1[['Référence', 'Titre', """Date d'inspection"""]].loc[general_euclidian_dist(df1[['Référence']].index[df1['Référence'].str.contains(option,case=False)],lda_output,n_result)])
    if dist == 'Cosine':
        st.table(df1[['Référence', 'Titre', """Date d'inspection"""]].loc[general_cosine_dist(df1[['Référence']].index[df1['Référence'].str.contains(option,case=False)],lda_output,n_result)])
    sim = df1[ 'Titre'].loc[general_cosine_dist(df1[['Référence']].index[df1['Référence'].str.contains(option,case=False)],lda_output,n_result)]
    option2 = st.selectbox('Documents similaires:', sim)

#Ajouter une conclusion (on a traité les lettres, identifié les topics, mis en place un système de recommandation, etc..
if page == "Suggestions nouveau document":
    st.title("Suggestions de nouveaux documents")
    suggest = st.text_input('Saisir les mots clefs', 'Essais, palonnier, etc.')
    st.write('Extension de requête par Word2Vec')
    uploaded_file = st.file_uploader("Importer un pdf", type="pdf")


    st.header("test html import")

    HtmlFile = open("lda_best_total.html.html", 'r', encoding='utf-8')
    source_code = HtmlFile.read()
    print(source_code)
    components.html(source_code)

