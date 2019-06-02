# -------------------------------------------------------------------------------------------------------------------------------------------------------------
# The our class for PDF files
# @Authors:  Alexey Titov and Shir Bentabou
# @Version: 1.0
# -------------------------------------------------------------------------------------------------------------------------------------------------------------

# libraries
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.pdfpage import PDFPage
from pdfminer.converter import XMLConverter, HTMLConverter, TextConverter
from pdfminer.layout import LAParams
try:
    from StringIO import StringIO
except ImportError:
    from io import StringIO
import matplotlib.pyplot as plt
import seaborn as sns
from pdf2image import convert_from_path
from PyPDF2 import PdfFileReader
import imutils
import cv2
import pytesseract
import sys

# fix UnicodeEncodeError
if sys.version[0] == '2':
    reload(sys)
    sys.setdefaultencoding("utf-8")


class readPDF:
    # dictionary for translate PDF language to tessaract language
    __lan_lst = {
        "en-us": "eng",
        "en": "eng",
        "en-za": "eng",
        "en-gb": "eng",
        "en-in": "eng",
        "es-co": "spa",
        "es": "spa",
        "de-de": "deu",
        "fr-fr": "fra",
        "fr-ca": "fra"}

    # dictionary for /Root/Lang 1 - except; 2 - a file have not /Root/Lang; 3
    # - /Root/Lang = ''; 4 - language
    __ans_list = dict()

    # constructor
    def __init__(self, ans_list):
        self.__ans_list = ans_list

    # this function read information from image
    def extract_text_image(self, imgPath):
        # Define config parameter
        # '--oem 1' for using LSTM OCR Engine
        config = ('--oem 1 --psm 3')

        # Read image from disk
        img = cv2.imread(imgPath, cv2.IMREAD_COLOR)

        # Read /Root/Lang
        values = self.__ans_list.get(imgPath)
        try:
            if (values[0] == 4):
                langs = value[1]
                imagetext = pytesseract.image_to_string(img, lang=langs, config=config)
            else:
                imagetext = pytesseract.image_to_string(img, config=config)
            return imagetext
        except Exception as ex:
            print(imgPath)
            print(ex)
            imagetext = "except"
            return imagetext

    # this function extract text from pdf
    def extractTEXT(self, filename, imagename):
        pagenos = set()
        data = ""
        fp = open(filename, 'rb')
        rsrcmgr = PDFResourceManager()
        retstr = StringIO()
        laparams = LAParams()
        device = TextConverter(rsrcmgr, retstr, codec='utf-8', laparams=laparams)
        # create a PDF interpreter object
        interpreter = PDFPageInterpreter(rsrcmgr, device)
        # Process each page contained in the document
        try:
            for page in PDFPage.get_pages(fp, pagenos, maxpages=0, password="", caching=True, check_extractable=False):
                interpreter.process_page(page)
                data = retstr.getvalue()
                # number of character is anomaly
                if (len(data) < 2 or len(data) > 60000):
                    data = self.extract_text_image(imagename)
                break
        except Exception as ex:
            print(ex)
            print(filename)
            data = self.extract_text_image(imagename)
        # Cleanup
        device.close()
        retstr.close()
        return data
