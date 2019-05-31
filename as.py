# -------------------------------------------------------------------------------------------------------------------------------------------------------------
# Main
# @Authors:  Alexey Titov and Shir Bentabou
# @Version: 1.0
# @Date 05.2019
# -------------------------------------------------------------------------------------------------------------------------------------------------------------

# libraries
from classes.dataPDF import dataPDF
from classes.createDATA import createDATA
from classes.readPDF import readPDF
import os
import sys
import csv
import argparse
import tempfile


if __name__ == "__main__":
    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--dataset", required=True,
                    help="path to input dataset")
    args = vars(ap.parse_args())
    # define the name of the directory to be created
    path_IMAGES = "IMAGES"
    path_TEXTS = "TEXTS"

    # create folders for images and texts
    try:
        os.mkdir(path_IMAGES)
        os.mkdir(path_TEXTS)
    except OSError:
        print("[!] Creation of the directories {} or {} failed, maybe the folders are exist".format(
            path_IMAGES, path_TEXTS))
    else:
        print(
            "[*] Successfully created the directories {} and {} ".format(path_IMAGES, path_TEXTS))
    folder_path = os.getcwd()
    dataset_path = os.path.join(folder_path, args["dataset"])

    # check if a folder of data is exist
    if (not os.path.exists(dataset_path)):
        print("[!] The {} folder is not exist!\n    GOODBYE".format(dataset_path))
        sys.exit()
 
    # create csv file
    with open("pdfFILES.csv", 'w') as csvFile:
            fields = ['File', 'Text']
            writer = csv.DictWriter(csvFile, fieldnames = fields)
            writer.writeheader()
    csvFile.close()
    # start create data
    print("+++++++++++++++++++++++++++++++++++ START CREATE DATA +++++++++++++++++++++++++++++++++++")
    obj_data = createDATA(folder_path, args["dataset"])

    # convert first page of pdf file to image
    result = obj_data.convert(dataset_path)
    if (result):
        print("[*] Succces convert pdf files")
    else:
        print("[!] Whoops. something wrong dude. enable err var to track it")
        sys.exit()

    # extract JavaScript from pdf file
    result = obj_data.extract(dataset_path)
    if (result):
        print("[*] Succces extract JavaScript from pdf files")
    else:
        print("[!] Whoops. something wrong dude. enable err var to track it")
        sys.exit()
    print("\n+++++++++++++++++++++++++++++++++++++++++ FINISH ++++++++++++++++++++++++++++++++++++++++\n")
    
    # start create vectors
    print("++++++++++++++++++++++++++++++++++ START CREATE VECTORS +++++++++++++++++++++++++++++++++")
    # dir of folder and filter for pdf files
    files = [f for f in os.listdir(dataset_path) if os.path.isfile(
        os.path.join(dataset_path, f))]
    files = list(filter(lambda f: f.endswith(('.pdf', '.PDF')), files))

    # variables for print information
    cnt_files = len(files)
    obj_pdfs = []
    obj_read = readPDF(obj_data.getDict())
    # loop over the input pdfs
    for (i, pdfFILE) in enumerate(files):
        # create pdf object
        obj_pdf = dataPDF(pdfFILE, folder_path+'/', args["dataset"])
        obj_pdf.calculate_histogram_blur()
        obj_pdf.calculate_dsurlsjsentropy()
        obj_pdf.save_text(obj_read.extractTEXT(obj_pdf.getFilename(), obj_pdf.getImage()))
        obj_pdfs.append(obj_pdf)
        # show an update every 50 pdfs
        if (i > 0 and i % 50 == 0):
            print("[INFO] processed {}/{}".format(i, cnt_files))
    print("[INFO] processed {}/{}".format(cnt_files, cnt_files))
    print("\n+++++++++++++++++++++++++++++++++++++++++ FINISH ++++++++++++++++++++++++++++++++++++++++\n")
 
    # start machine learning
    print("+++++++++++++++++++++++++++++++++ START MACHINE LEARNING ++++++++++++++++++++++++++++++++")
    print(len(obj_pdfs))



