# -------------------------------------------------------------------------------------------------------------------------------------------------------------
# Main, union vectors
# @Authors:  Alexey Titov and Shir Bentabou
# @Version: 1.0
# @Date 05-06.2019
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
import numpy as np
from numpy import random
# machine learning libraries
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition.truncated_svd import TruncatedSVD
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn import metrics
from sklearn.pipeline import Pipeline, make_pipeline
# import AdaBoostClassifier
from sklearn.ensemble import AdaBoostClassifier
# import AdaBoostRegressor
from sklearn.ensemble import AdaBoostRegressor
# import XGBClassifier
from xgboost import XGBClassifier
# import XGBRegressor
from xgboost.sklearn import XGBRegressor

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
    labels = []
    obj_read = readPDF(obj_data.getDict())
    # loop over the input pdfs
    for (i, pdfFILE) in enumerate(files):
        label = -1
        if ("mal" == pdfFILE.split(".")[0]):
           label = 1
        else:
           label = 0
        labels.append(label)
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
 
    # start union vectors 
    print("+++++++++++++++++++++++++++++++++ START UNION VECTORS ++++++++++++++++++++++++++++++++")
    labels = np.array(labels)
    my_tags = ['0','1']
    # partition the data into training and testing splits, using 80%
    # of the data for training and the remaining 20% for testing
    (trainF, testF, trainLabels, testLabels) = train_test_split(obj_pdfs, labels, test_size = 0.20, random_state = 42)
    
    # text for train
    trainForVector = []
    for pdf in trainF:
        trainForVector.append(pdf.getText())

    # text for test
    for pdf in testF:
        trainForVector.append(pdf.getText())

    # strip_accents = 'unicode' : replace all accented unicode char ;  use_idf = True : enable inverse-document-frequency reweighting ;
    # smooth_idf = True : prevents zero division for unseen words
    tfidf_vect= TfidfVectorizer(strip_accents = 'unicode', use_idf = True, smooth_idf = True, sublinear_tf = False)
    trainForVector = tfidf_vect.fit_transform(trainForVector)
    num_features = len(tfidf_vect.get_feature_names())
    # n_components : Desired dimensionality of output data. Must be strictly less than the number of features.
    # n_iter : Number of iterations for randomized SVD solver. 
    # random_state : If int, random_state is the seed used by the random number generator.
    pca = TruncatedSVD(n_components = num_features-1, n_iter = 7, random_state = 42)
    trainForVector = pca.fit_transform(trainForVector)

    # train
    i = 0
    trainFeat = []
    for pdf in trainF:
        v_all = list(pdf.getImgHistogram()) + list(trainForVector[i]) + list(pdf.getFeatVec())
        trainFeat.append(v_all)
        i += 1
    # test
    testFeat = []
    for pdf in testF:
        v_all = list(pdf.getImgHistogram()) + list(trainForVector[i]) + list(pdf.getFeatVec())
        testFeat.append(v_all)
        i += 1
    trainFeat = np.array(trainFeat)
    testFeat = np.array(testFeat)
    print("\n+++++++++++++++++++++++++++++++++++++++++ FINISH ++++++++++++++++++++++++++++++++++++++++\n")

    # start boost
    print("+++++++++++++++++++++++++++++++++++++++ START BOOST +++++++++++++++++++++++++++++++++++++")
    # instantiating AdaBoostClassifier
    abc = AdaBoostClassifier(n_estimators = 100, random_state = 0)
    abc.fit(trainFeat, trainLabels)
    print("Feature importances for AdaBoostClassifier: ")
    print(abc.feature_importances_)
    # make predictions for test data
    predictions = abc.predict(testFeat)
    accuracy = accuracy_score(testLabels, predictions)
    print("Accuracy of AdaBoostClassifier: %.2f%%" % (accuracy * 100.0))

    # instantiating AdaBoostRegressor (similar to logistic regression)
    abr = AdaBoostRegressor(random_state = 0, n_estimators = 100)
    abr.fit(trainFeat, trainLabels)
    print("Feature importances for AdaBoostRegressor: ")
    print(abr.feature_importances_)
    # make predictions for test data
    predictions = abr.predict(testFeat)
    accuracy = accuracy_score(testLabels, predictions)
    print("Accuracy of AdaBoostRegressor: %.2f%%" % (accuracy * 100.0))

    # instantiating XGBClassifier
    xgbc = XGBClassifier()
    xgbc.fit(trainFeat, trainLabels)
    print("Feature importances for XGBClassifier: ")
    print(xgbc.feature_importances_)
    # make predictions for test data
    predictions = xgbc.predict(testFeat)
    accuracy = accuracy_score(testLabels, predictions)
    print("Accuracy of XGBClassifier: %.2f%%" % (accuracy * 100.0))

    # instantiating XGBRegressor (similar to linear regression)
    xgbr = XGBRegressor(n_estimators = 100, max_depth = 3)
    xgbr.fit(trainFeat, trainLabels)
    print("Feature importances for XGBRegressor: ")
    print(xgbr.feature_importances_)
    # make predictions for test data
    predictions = xgbr.predict(testFeat)
    accuracy = accuracy_score(testLabels, predictions.round())
    print("Accuracy of XGBRegressor: %.2f%%" % (accuracy * 100.0))
    print("\n+++++++++++++++++++++++++++++++++++++++++ FINISH ++++++++++++++++++++++++++++++++++++++++\n")

