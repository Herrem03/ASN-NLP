from bs4 import BeautifulSoup
import requests
from lxml import html
import json
import re
import numpy as np

ref = []
n_pages = 851             #Aller vérifier sur la page de l'ASN
dates = []
titres = []
links = []
refs = []
dates_cleared = []
titres_cleared = []
page_erreur = []
a=[]

def pause():
    Pause = input("Appuyer sur entrée pour continuer...")
    return Pause

#iterate over pages
for i in range(1, n_pages):
    dates_cleared = []
    titres_cleared = []
    count = 0
    print('Current page :', format(i))
    page = requests.get("https://www.asn.fr/Controler/Actualites-du-controle/Lettres-de-suite-d-inspection-des-installations-nucleaires/(searchText)/epr/(page)/{}".format(i))
    soup = BeautifulSoup(page.content, 'html.parser')
    tree = html.fromstring(page.content)
    soup.prettify()
    it = 1
    for link in soup.findAll('a', attrs={'href': re.compile("^https://")}):
        if link.get('href').find('https://www.asn.fr/Controler/Actualites-du-controle/Lettres-de-suite-d-inspection-des-installations-nucleaires/') != -1:
            a.append(link.get('href'))

    for element in tree.xpath('//p[@class="Teaser-infos Teaser-infos--yellow"]/text()'):
        if tree.xpath('//*[@id="content"]/div[6]/div[{}]/a[2]/div[2]/p[1]/text()'.format(it)) != []:
            refs = refs + tree.xpath('//*[@id="content"]/div[6]/div[{}]/a[2]/div[2]/p[1]/text()'.format(it))
            it += 1
        else:
            if tree.xpath('//*[@id="content"]/div[6]/div[{}]/a[2]/div[2]/p[2]/text()'.format(it)) != []:
                refs = refs + tree.xpath('//*[@id="content"]/div[6]/div[{}]/a[2]/div[2]/p[2]/text()'.format(it))
                it += 1
            else:
                count += 1
                if dates_cleared == [] and titres_cleared == []:
                    dates_cleared = tree.xpath('//p[@class="Teaser-infos Teaser-infos--yellow"]/text()')
                    titres_cleared = tree.xpath('//a[@class="Teaser-title"]/text()')
                del dates_cleared[it - count]
                del titres_cleared[it - count]
                del a[it-count]
                it += 1

    if dates_cleared == [] and titres_cleared ==[]:
        dates = dates + tree.xpath('//p[@class="Teaser-infos Teaser-infos--yellow"]/text()')
        titres = titres + tree.xpath('//a[@class="Teaser-title"]/text()')
        links = links + a
        a = []
    else:
        print(titres_cleared)
        dates = dates + dates_cleared
        titres = titres + titres_cleared
        links = links + a
        a = []

    print('dates :', np.size(dates))
    print('titres :', np.size(titres))
    print('refs :', np.size(refs))
    print('liens :', np.size(links))
    print('#-----------------#')
    if np.size(dates) != np.size(refs):
        page_erreur.append(i)

#Create dictionary
dict = {1: refs, 2: titres, 3: dates, 4: links}
print(dict)
print('Erreurs détectées aux pages suivantes :', page_erreur)

#Save dictionary
with open('ASN_scrapped_Flammanville', 'w+') as json_file:
    jsoned_data = json.dumps(dict, indent=True, ensure_ascii=False)
    json_file.write(jsoned_data)

print('Traitement terminé')
