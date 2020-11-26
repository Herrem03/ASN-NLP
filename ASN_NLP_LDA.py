import numpy as np
import pandas as pd
import re, nltk, spacy, gensim
import os
from time import time

# Sklearn
from sklearn.decomposition import LatentDirichletAllocation, TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.manifold import TSNE
from sklearn.model_selection import GridSearchCV
from pprint import pprint

# Plotting tools
import pyLDAvis.sklearn
import seaborn as sns

# Bokeh
from bokeh.plotting import figure, show, output_file, save
from bokeh.models import HoverTool, CustomJS, ColumnDataSource, Slider
from bokeh.layouts import column
from bokeh.palettes import all_palettes
#Data
import json


def pause():
    Pause = input("Appuyer sur entrée pour continuer...")
    return Pause

os.getcwd()
df = pd.read_json('ASN_NLP_dict')
print(df.head(15))
print(df)
# Convert to list
data = df['Contenu'].values.tolist()
dates = df['Titres'].values.tolist()
#LDA parameters
n_topics = 35

# Remove Emails
data = [re.sub('\S*@\S*\s?', '', sent) for sent in data]

# Remove new line characters
data = [re.sub('\s+', ' ', sent) for sent in data]

# Remove distracting single quotes
data = [re.sub("\'", "", sent) for sent in data]

def sent_to_words(sentences): #A virer et remplacer par spacy nlp.pipe
    for sentence in sentences:
        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))  # deacc=True removes punctuations

data_words = list(sent_to_words(data))

def lemmatization(texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    texts_out = []
    for sent in texts:
        doc = nlp(" ".join(sent))
        texts_out.append(" ".join([token.lemma_ for token in doc if token.pos_ in allowed_postags])) #virer if token.lemma_ not in ['-PRON-'] else ''
    return texts_out

nlp = spacy.load('fr_core_news_md')

# Do lemmatization keeping only Noun, Adj, Verb, Adverb
data_lemmatized = lemmatization(data_words, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])
print(data_lemmatized)
#pause()

max_df = [.5, .6, .7 ,.8]

for element in max_df:
    t1 = time()
    vectorizer = CountVectorizer(analyzer='word',
                                 min_df=10,  # minimum read occurences of a word
                                 ngram_range=(1, 2),  # compute bigrams
                                 max_df=element,  # overall frequency taken in account 50%
                                 token_pattern='[a-zA-Z]{4,}',  # num chars > 4
                                 encoding='utf-8',
                                 strip_accents='ascii',
                                 decode_error='strict'
                                 # max_features=50000,              # max number of uniq words
                                 )
    data_vectorized = vectorizer.fit_transform(data_lemmatized)

    # Define Search Param
    search_params = {'n_components': [10, 15, 20, 25, 30], 'learning_decay': [.5, .6, .7, .8]}

    # Init the Model
    lda = LatentDirichletAllocation()

    # Init Grid Search Class
    model = GridSearchCV(lda, param_grid=search_params)

    # Do the Grid Search
    model.fit(data_vectorized)

    # Best Model
    best_lda_model = model.best_estimator_

    # Model Parameters
    print("Best Model's Params: ", model.best_params_)

    # Log Likelihood Score
    print("Best Log Likelihood Score: ", model.best_score_)

    # Perplexity
    print("Model Perplexity: ", best_lda_model.perplexity(data_vectorized))

    # lda_output = lda_model.fit_transform(data_vectorized)

    # PyLDAVis
    panel = pyLDAvis.sklearn.prepare(best_lda_model, data_vectorized, vectorizer, mds='tsne')
    pyLDAvis.save_html(panel, 'LDA_v2_Gridsearch_unigram_max_df{0}_topics_10to30.html'.format(element))

    df_cv_results = pd.DataFrame(model.cv_results_)
    df_cv_results.to_csv("LDAGridSearchResults_max_df_{}.csv".format(element), header=True, index=False, encoding='utf-8')

    plot = sns.pointplot(x="param_n_components", y="mean_test_score", hue="param_learning_decay", data=df_cv_results)
    fig = plot.get_figure()
    fig.savefig("output_max_df_{}.png".format(element))
    print('Time to perform max_df = {0}: {1} mins'.format(element, round((time() - t1) / 60, 2)))


lda_output = lda_model.fit_transform(data_vectorized)
np.savetxt("LDA_output_test1.csv", lda_output, delimiter=",")

with open('LDA_output_test1', 'w+') as json_file:
    jsoned_data = json.dumps(lda_output, indent=True, ensure_ascii=False)
    json_file.write(jsoned_data)
pause()

from joblib import dump, load
import pickle

#dump(lda_model, 'test1_LDA.joblib')
with open("test1_LDA.pickle", "wb") as f:
    pickle.dump(lda_model, f,protocol=2)

with open("vectorizer_test1_LDA.pickle", "wb") as f:
    pickle.dump(vectorizer, f,protocol=2)


def show_topics(vectorizer=vectorizer, lda_model=lda_model, n_words=20):
    keywords = np.array(vectorizer.get_feature_names())
    topic_keywords = []
    for topic_weights in lda_model.components_:
        top_keyword_locs = (-topic_weights).argsort()[:n_words]
        topic_keywords.append(keywords.take(top_keyword_locs))
    return topic_keywords

topic_keywords = show_topics(vectorizer=vectorizer, lda_model=lda_model, n_words=15)

# Topic - Keywords Dataframe
df_topic_keywords = pd.DataFrame(topic_keywords)
df_topic_keywords.columns = ['Word '+str(i) for i in range(df_topic_keywords.shape[1])]
df_topic_keywords.index = ['Topic '+str(i) for i in range(df_topic_keywords.shape[0])]
print(df_topic_keywords)


#TSNE
tsne_model = TSNE(n_components  =2, perplexity =30, verbose  =1, random_state  =0, angle  =.99, init ='pca')
tsne_lda  = tsne_model.fit_transform(lda_output)
tsne_lda = pd.DataFrame(tsne_lda, columns=['x','y'])
tsne_lda['hue'] = lda_output.argmax(axis=1)
print(lda_output)
pause()

#Vis' TSNE
my_colors = [(all_palettes['Category20'][20] + all_palettes['Category20'][20])[i] for i in tsne_lda.hue]
source = ColumnDataSource(
        data=dict(
            x = tsne_lda.x,
            y = tsne_lda.y,
            colors = my_colors,
            reference = df[1],
            titre = df[2],
            date = df[3],
            alpha = [0.9] * tsne_lda.shape[0],
            size = [7] * tsne_lda.shape[0]
        )
    )

hover_tsne = HoverTool(names=["df"], tooltips="""
    <div style="margin: 10">
        <div style="margin: 0 auto; width:300px;">
            <span style="font-size: 12px; font-weight: bold;">Référence:</span>
            <span style="font-size: 12px">@reference</span>
            <span style="font-size: 12px; font-weight: bold;">Titre:</span>
            <span style="font-size: 12px">@titre</span>
            <span style="font-size: 12px; font-weight: bold;">Date d'inspection:</span>
            <span style="font-size: 12px">@date</span>
        </div>
    </div>
    """)

tools_tsne = [hover_tsne, 'pan', 'wheel_zoom', 'reset']
plot_tsne = figure(plot_width=700, plot_height=700, tools=tools_tsne, title='TSNE Lettres inspection ASN')
plot_tsne.circle('x', 'y', size='size', fill_color='colors',
                     alpha='alpha', line_alpha=0, line_width=0.01, source=source, name="df")

callback = CustomJS(args=dict(source=source), code=
    """
    var data = source.data;
    var f = cb_obj.value
    x = data['x']
    y = data['y']
    colors = data['colors']
    alpha = data['alpha']
    title = data['reference']
    year = data['titre']
    date = data['date']
    size = data['size']
    for (i = 0; i < x.length; i++) {
        if (year[i] <= f) {
            alpha[i] = 0.9
            size[i] = 7
        } else {
            alpha[i] = 0.05
            size[i] = 4
        }
    }
    source.change.emit();
    """)

layout = column(#slider,
                plot_tsne)
output_file("TSNE_ASN_utf8_data_topics35_perplexity30_maxdf005_lem.html")
save(layout)

