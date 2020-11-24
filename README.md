<h1>ASN-NLP</h1>

*:warning: This is not the production version and is only for reference and learning purposes.*

Streamlit app analyzing French Nuclear Regulation Authority's inspection letters using LDA, Word2vec and text mining techniques. 

<h2>About</h2>
Regulatory control in the french nuclear industry has an impact on project's costs and planning
This project aims to :
- Build a content-based recommendation app of french nuclear regulation authority's inspection letters 
- Analyze French nuclear regulation

See Medium stories for more details :

<div>
 
[Part I | Create an exploratory app using LDA, Word2vec and Streamlit](https://medium.com/@remi.martinie03/corpus-analysis-using-nlp-a-glimpse-at-french-nuclear-regulation-ce84697d47bf)

[Part II | Extracting knowledge from French Nuclear Safety Authority's Inspection letters](https://medium.com/@remi.martinie03/corpus-analysis-using-nlp-a-glimpse-at-french-nuclear-regulation-482ee6288b12)

<h2>Requirements</h2>
<div>

<li>[BeautifulSoup](https://www.crummy.com/software/BeautifulSoup/bs4/doc/) | Library for webscrapping

[PDFMiner](https://pypi.org/project/pdfminer/) | Text extraction tool for PDF documents

[Scikit-learn](https://scikit-learn.org/stable/modules/classes.html) | ML package used here for Latent Dirichlet Allocation and GridSearchCV

[Gensim](https://radimrehurek.com/gensim/auto_examples/index.html#documentation) | NLP package used here for Word2Vec

[SpaCy](https://spacy.io/api) | NLP package for text processing (stop words, lemmatization, punctuation, tokenization, etc.)

[Streamlit](https://docs.streamlit.io/en/stable/) | Library to create and share custom web apps for machine learning and data science.

[pyLDAvis](https://github.com/bmabey/pyLDAvis) | Interactive topic modeling visualization library

[Plotly](https://plotly.com/graphing-libraries/) | Graphing library

[Pandas](https://pandas.pydata.org/docs/) | Data handling and analysis library</li>

<h2>Dataset</h2>

Dataset is generated using :
<li>Download_pdf.py</li>
<li>Conversion_pdf. py</li>

:warning: Some .pdf are scans and therefore cannot be used. About 97% of inspection letters are taken in account up to this date : 23/11/2020.

<h2>Latent Dirichlet Allocation</h2>
LDA is used to build a recommendation system.

ASN_LDA.py includes :
- Text preprocessing : removing stopwords, remove punctutation, lemmatization
- LDA pipeline for enhanced GridSearch over *'max_df'* in CountVectorizer(), *'learning_decay'* and *'n_components'* in LatentDirichletAllocation()

<h2>Word2Vec</h2>

<h2>Text mining</h2>

ASN_mining.py includes :
- Demands analysis 
- Knowledge graph 

<h2>Streamlit App</h2>
App is available in French & English (ASN_app_fr.py & ASN_app_en.py)

ASN_app_fr.py & ASN_app_en.py includes :
- Data visualization of LDA model, Word2Vec model and text mining analysis
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
