# Phase5
The fifth phase of our project. In this phase we are hoping to create a final classifier of PDF files.
The final classifier will be based on the three previous machines in our project: image, text, and features.
The process of the final machine will be the following:
  * install: `sudo pip3 install xgboost`
  * Extract all data needed for the three base machines (image, text, features) - this is done using classes, imported into as.py.
  * Create base vectors for every sample (image, text, features)
  * Run every base machine on the samples, and return the calssification of the sample by every machine.
  * Create a vector for the boost algorithm from the base machines classifications for every sample.
  * Run boost algorithm with RF on sample boost vectors.
  * Return boost algorithm accuracy.

Hello darkness my old friend

you need to install python2 - and then install a shit ton of libraries as well.

This thing runs on python3 but calls python2 on several occasions.

You also need python3. you should find a way so python redirects to python2 and python3 does python3 as usual. if you fail to do so, edit the classes (createDATA, etc...) and manually set the 'python' command to 'python2'

python2 -m pip install opencv-python==4.2.0.32

nodejs -v
v8.10.0

npm -v
3.5.2

in jast, in folder js, in is_js.py - change nodejs to node if you used nvm to install nodejs (fucking shit)...


pip2 install pdfminer==20140328

pip3 install pdfminer.six==20181108


In ExtractJS.txt:
replace: 
extract js > /home/tzar/Desktop/Final_Project/phase5/JSfromPDF.txt

with:
extract js > /newdrive/home/tzar/Desktop/Final_Project/shir_test/JSfromPDF.txt


https://stackoverflow.com/questions/10592605/save-classifier-to-disk-in-scikit-learn
