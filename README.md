<h1>ASN-NLP</h1>

*Disclaimer : This is not the production version and is only for reference and learning purposes.*

Web app analyzing French Nuclear Regulation Authority's inspection letters using LDA, Word2vec and Streamlit. 

<h2>Requirements</h2>

<h2>Dataset</h2>

Dataset is generated using :
<li>Download_pdf.py</li>
<li>Conversion_pdf. py</li>

<h2>Latent Dirichlet Allocation</h2>
LDA is used for text clustering to build a recommendation system.

ASN_LDA.py includes :
- Text preprocessing : removing stopwords, remove punctutation, lemmatization
- LDA pipeline for enhanced GridSearch over *'max_df'* in CountVectorizer(), *'learning_decay'* and *'n_components'* in LatentDirichletAllocation()


<h2>Streamlit App</h2>

<h2>Author</h2>
[Rémi MARTINIE](www.linkedin.com/in/rémi-martinie-3107291b9)


<h2>Acknowledgments</h2>
Selva Prabhakaran from Machine Learning Plus for the great tutorials [https://www.machinelearningplus.com/category/nlp/]
