import pandas as pd
import os
import requests
import json
import numpy as np
import time
import os.path

#Extraire le texte du fichier pdf
from pdfminer.pdfparser import PDFParser
from pdfminer.pdfdocument import PDFDocument
from pdfminer.pdfpage import PDFPage
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.pdfpage import PDFTextExtractionNotAllowed
from pdfminer.layout import LAParams, LTTextBox, LTTextLine
from pdfminer.converter import PDFPageAggregator

#Nettoyage données
from nltk.tokenize import word_tokenize

def pause():
    Pause = input("Appuyer sur entrée pour continuer...")
    return Pause

os.getcwd()
df = pd.read_json('ASN_scrapped_Flammanville')

#----------------------------------
liens = df[4].values.tolist()
dates = df[3].values.tolist()
titres = df[2].values.tolist()
refs = df[1].values.tolist()
content = []
i = 0
count = 0
error_files = []

for x in liens:
    seconds_1 = time.time()
    extracted_text = ""
    url = liens[i]
    print(refs[i])
    target = '/home/herrem/Documents/ASN/ASN_pdf/{}'.format(refs[i])
    if os.path.isfile(target):
        print('Fichier déjà téléchargé')
    else:
        try:
            myfile = requests.get(url)
        except:
            print("Nouvelle tentative de connexion dans 30 s")
            time.sleep(30)
            myfile = requests.get(url)
        open(target, 'wb').write(myfile.content)
        print("Fichier absent, téléchargement effectué")
    fp = open(target, 'rb')
    parser = PDFParser(fp)
    try:
        document = PDFDocument(parser)
        if not document.is_extractable:
            raise PDFTextExtractionNotAllowed
        rsrcmgr = PDFResourceManager()
        laparams = LAParams()
        device = PDFPageAggregator(rsrcmgr, laparams=laparams)
        interpreter = PDFPageInterpreter(rsrcmgr, device)

        for page in PDFPage.create_pages(document):
            interpreter.process_page(page)
            layout = device.get_result()
            for lt_obj in layout:
                if isinstance(lt_obj, LTTextBox) or isinstance(lt_obj, LTTextLine):
                    extracted_text += lt_obj.get_text()
        fp.close()
        print("Pdf extrait")
        tokens = word_tokenize(extracted_text)
        tokens = [x.lower() for x in tokens]
        words_str = ' '.join(tokens)
        content.append(words_str)
        print(words_str)
        i += 1
        seconds_2 = time.time()

        print("Temps de traitement du fichier =", round(seconds_2-seconds_1,3),"s")
        print("Nombre de fichiers traités : ", i)
        print("Traitement terminé à ", round(i/np.size(liens)*100, 3), "%")
        print("#-----------------------------#")
    except:
        content.append('fichier vide')
        error_files.append(refs[i])
        print('Erreur dans extraction, fichier ignoré :')
        count += 1
        i += 1
        seconds_2 = time.time()

#A TRANSFERER DANS extraction pdf !
print("Nombre de fichiers ignorés :", count)
print('Fichiers ignorés :', error_files)

i = 0
for element in content:
    content[i].encode("utf-8")
    refs[i].encode("utf-8")
    titres[i].encode("utf-8")
    dates[i].encode("utf-8")
    liens[i].encode("utf-8")
    i+=1

dict_LDA = {1 : refs, 2: titres, 3: dates, 4: liens, 5: content} #ajouter refs

with open('ASN_LDA_dict_flammanville_utf8', 'w+') as json_file:
    jsoned_data = json.dumps(dict_LDA, indent=True, ensure_ascii=False)
    json_file.write(jsoned_data)
