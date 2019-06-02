# -------------------------------------------------------------------------------------------------------------------------------------------------------------
# Main
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
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn import metrics
from sklearn.pipeline import Pipeline, make_pipeline
# importing K-Means
from sklearn.cluster import KMeans
# importing KNN
from sklearn.neighbors import KNeighborsClassifier
# import RF
from sklearn.ensemble import RandomForestClassifier
# import SVM
from sklearn.linear_model import SGDClassifier
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
    # argument for KNN
    ap.add_argument("-k", "--neighbors", type = int, default = 5,
		help="# of nearest neighbors for classification")
    # arguments for k-means-clustering
    ap.add_argument("-c", "--clusters", type = int, default = 5,
		help="the number of clusters to form as well as the number of centroids to generate")
    ap.add_argument("-j", "--jobs", type = int, default = -1,
		help="the number of jobs to use for the computation. ")
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
 
    # start machine learning
    print("+++++++++++++++++++++++++++++++++ START MACHINE LEARNING ++++++++++++++++++++++++++++++++")
    labels = np.array(labels)
    my_tags = ['0','1']
    # partition the data into training and testing splits, using 50%
    # of the data for training and the remaining 50% for testing
    (trainF, testF, trainLabels, testLabels) = train_test_split(obj_pdfs, labels, test_size = 0.50, random_state = 42)
    trainFeat = []
    testFeat = []
    for pdf in trainF:
        trainFeat.append(pdf.getImgHistogram())
    for pdf in testF:
        testFeat.append(pdf.getImgHistogram())
    trainFeat = np.array(trainFeat)
    testFeat = np.array(testFeat)

    # instantiating kmeans and knn
    km = KMeans(algorithm = 'auto', copy_x = True, init = 'k-means++', max_iter = 300, n_clusters = args["clusters"], n_init = 10, n_jobs = args["jobs"])
    knn = KNeighborsClassifier(algorithm = 'auto', n_neighbors = args["neighbors"], n_jobs = args["jobs"])

    # training knn model
    knn.fit(trainFeat, trainLabels)
    # testing knn
    predictions1_n = knn.predict(testFeat)

    # training km model
    km.fit(trainFeat)
    # testing km
    predictions1_m = km.predict(testFeat)

    # creating vector for Random Forest on features
    trainFeat = []
    testFeat = []
    for pdf in trainF:
        trainFeat.append(pdf.getFeatVec())
    for pdf in testF:
        testFeat.append(pdf.getFeatVec())
    trainFeat = np.array(trainFeat)
    testFeat = np.array(testFeat)
    # instantiating Random Forest
    ranfor = Pipeline([
        ('clf', RandomForestClassifier(n_estimators = 30, random_state = 0)),
    ])
    ranfor.fit(trainFeat, trainLabels)
    predictions3 = ranfor.predict(testFeat)

    # creating vector for SVM on text
    trainFeat = []
    testFeat = []
    for pdf in trainF:
        trainFeat.append(pdf.getText())
    for pdf in testF:
        testFeat.append(pdf.getText())
    # instantiating Linear Support Vector Machine
    sgd = Pipeline([('vect', CountVectorizer()),
                ('tfidf', TfidfTransformer()),
                ('clf', SGDClassifier(loss = 'hinge', penalty = 'l2', alpha = 1e-3, random_state = 42, max_iter = 200, tol = 1e-3)),
               ])
    sgd.fit(trainFeat, trainLabels)
    predictions2 = sgd.predict(testFeat)
    print("\n+++++++++++++++++++++++++++++++++++++++++ FINISH ++++++++++++++++++++++++++++++++++++++++\n")

    # start boost
    print("+++++++++++++++++++++++++++++++++++++++ START BOOST +++++++++++++++++++++++++++++++++++++")
    # creating vectors
    trainFeat = []
    for p1, p2, p3 in zip(predictions1_m, predictions2, predictions3):
        p_all = [p1, p2, p3]
        trainFeat.append(p_all)
    trainFeat = np.array(trainFeat)
    # partition the data into training and testing splits, using 60%
    # of the data for training and the remaining 40% for testing
    (trainFeat, testFeat, trainLabels, testLabels) = train_test_split(trainFeat, testLabels, test_size = 0.60, random_state = 42)

    # instantiating AdaBoostClassifier
    abc = AdaBoostClassifier(n_estimators = 100, random_state = 0)
    abc.fit(trainFeat, trainLabels)
    print("Feature importances for AdaBoostClassifier: ")
    print(abc.feature_importances_)
    # make predictions for test data
    predictions = abc.predict(testFeat)
    accuracy = accuracy_score(testLabels, predictions)
    print("Accuracy of AdaBoostClassifier: %.2f%%" % (accuracy * 100.0))

    # instantiating AdaBoostRegressor
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

    # instantiating XGBRegressor
    xgbr = XGBRegressor(n_estimators = 100, max_depth = 3)
    xgbr.fit(trainFeat, trainLabels)
    print("Feature importances for XGBRegressor: ")
    print(xgbr.feature_importances_)
    # make predictions for test data
    predictions = xgbr.predict(testFeat)
    accuracy = accuracy_score(testLabels, predictions.round())
    print("Accuracy of XGBRegressor: %.2f%%" % (accuracy * 100.0))
    print("\n+++++++++++++++++++++++++++++++++++++++++ FINISH ++++++++++++++++++++++++++++++++++++++++\n")

