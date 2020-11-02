# coding: utf-8
import os
from pdfminer.pdfparser import PDFParser
from pdfminer.pdfdocument import PDFDocument
from pdfminer.pdfpage import PDFPage
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.pdfpage import PDFTextExtractionNotAllowed
from pdfminer.layout import LAParams, LTTextBox, LTTextLine
from pdfminer.converter import PDFPageAggregator

count = 0
base_path = os.getcwd()

for filename in os.listdir(os.getcwd()):
    if filename.endswith(".pdf"):
        filename = filename.replace('.pdf', '')
        my_file = os.path.join(base_path + "/" + "{}.pdf".format(filename))
        log_file = os.path.join(base_path + "/" + "{}.txt".format(filename))
        extracted_text = ""
        fp = open(my_file, 'rb')
        parser = PDFParser(fp)
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
        with open(log_file, "w") as my_log:
            my_log.write(extracted_text)
        print("Fichier converti !")
        count += count

print('Nombre de fichiers convertis : ', count)




