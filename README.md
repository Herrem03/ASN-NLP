<h1>ASN-NLP</h1>

*:warning: This is not the production version and is only for reference and learning purposes.*

Streamlit app analyzing French Nuclear Regulation Authority's inspection letters using LDA, Word2vec and text mining techniques. 

<h2>About</h2>
This project aims to :
<li>Build a content-based recommendation app of french nuclear regulation authority's inspection letters</li>
<li>Get an overall understanding of how basic nuclear facilities are controlled in France</li>

<br>
See Medium story for more details :

<div>
 
[Create an exploratory app using LDA, Word2vec and Streamlit](https://medium.com/@remi.martinie03/corpus-analysis-using-nlp-a-glimpse-at-french-nuclear-regulation-ce84697d47bf)


<h2>Requirements</h2>
See requirements.txt for complete list of requirements. Main packages used are : <br>
<div>

[BeautifulSoup](https://www.crummy.com/software/BeautifulSoup/bs4/doc/) ==4.7.1 | Webscrapping library.

[PDFMiner](https://pypi.org/project/pdfminer/) ==20181108 | Text extraction tool for PDF documents.

[Scikit-learn](https://scikit-learn.org/stable/modules/classes.html) ==0.21.2 | ML package used here for Latent Dirichlet Allocation and GridSearchCV.

[Gensim](https://radimrehurek.com/gensim/auto_examples/index.html#documentation) ==3.7.3 | NLP package used here for Word2Vec.

[SpaCy](https://spacy.io/api) ==2.1.4 | NLP package for text processing (stop words, lemmatization, punctuation, tokenization, etc.).

[Streamlit](https://docs.streamlit.io/en/stable/) ==0.57.3 | Library to create and share custom web apps for machine learning and data science.

[pyLDAvis](https://github.com/bmabey/pyLDAvis) ==2.1.2 | Interactive topic modeling visualization library.

[Plotly](https://plotly.com/graphing-libraries/) ==4.6.0 | Graphing library.

[Pandas](https://pandas.pydata.org/docs/) ==0.24.2 | Data handling library.

<h2>Dataset</h2>

The dataset is larger than 200Mo and therefore not uploadable on GitHub. Use the following scripts to generate the dataset on your computer :
<li>ASN_NLP_Data.py</li>
<li>ASN_NLP_Scrap. py</li>

<h2>Latent Dirichlet Allocation</h2>
LDA is used to build a recommendation system.

ASN_NLP_LDA.py includes :
- Text preprocessing : removing stopwords, remove punctutation, lemmatization
- LDA pipeline for enhanced GridSearch over : *'max_df'* in CountVectorizer(), *'learning_decay'* in LatentDirichletAllocation(), *'n_components'* in LatentDirichletAllocation()

<h2>Word2Vec</h2>

ASN_NLP_Word2vec.py

<h2>Streamlit App</h2>
App is available only in French

ASN_NLP_App.py includes :
- Basic search engine
- Recommendation system (from user query extended by Word2Vec or .pdf file)

<h2>Author</h2>
<div>
  
[Rémi MARTINIE](https://www.linkedin.com/in/rémi-martinie-3107291b9/foo)

</div>
<h2>Contribution</h2>
If you want to add features/improvement or report issues, feel free to send a pull request !

<h2>Acknowledgments</h2>
<div> 
  
[Machine Learning Plus](http://www.machinelearningplus.com/category/nlp/) 

</div>
