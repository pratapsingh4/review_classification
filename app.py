
from flask import Flask,render_template,url_for,request
import os
import fnmatch
from textblob import TextBlob
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from nltk import pos_tag,pos_tag_sents
import regex as re
import operator
from sklearn.svm import SVC, LinearSVC
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.cross_validation import train_test_split  
from sklearn import metrics
from sklearn import svm
from sklearn.grid_search import GridSearchCV
import pickle
from nltk.corpus import stopwords
from nltk.corpus.reader.wordnet import WordNetError




app = Flask(__name__)

@app.route('/')
def home():
	return render_template('home.html')

@app.route('/predict',methods=['POST'])

	# df= pd.read_csv("YoutubeSpamMergedData.csv")
	# df_data = df[["CONTENT","CLASS"]]
	# # Features and Labels
	# df_x = df_data['CONTENT']
	# df_y = df_data.CLASS
 #    # Extract Feature With CountVectorizer
	# corpus = df_x
	# cv = CountVectorizer()
	# X = cv.fit_transform(corpus) # Fit the Data
	# from sklearn.model_selection import train_test_split
	# X_train, X_test, y_train, y_test = train_test_split(X, df_y, test_size=0.33, random_state=42)
	# #Naive Bayes Classifier
	# from sklearn.naive_bayes import MultinomialNB
	# clf = MultinomialNB()
	# clf.fit(X_train,y_train)
	# clf.score(X_test,y_test)
	# #Alternative Usage of Saved Model
	# # ytb_model = open("naivebayes_spam_model.pkl","rb")
	# # clf = joblib.load(ytb_model)
def predict():
	global os

	path ='D:/Software_Engineering/ML APP/data/op_spam_v1.4'
	label =[]
	configfiles =[os.path.join(subdir,f)
	for subdir,dirs,files in os.walk(path)
		for f in fnmatch.filter(files,'*.txt')]

	for f in configfiles:
		c=re.search('(truth|deceptive)\w',f)
		label.append(c.group())

	labels=pd.DataFrame(label,columns=['Labels'])

	
	review = []
	directory =os.path.join(path)
	for subdir,dirs ,files in os.walk(directory):
	   # print (subdir)
		for file in files:
			if fnmatch.filter(files, '*.txt'):
				f=open(os.path.join(subdir, file),'r')
				a = f.read()
				review.append(a)

	reviews=pd.DataFrame(review,columns=['Reviews'])
	result=pd.merge(reviews,labels,right_index=True,left_index=True)
	

	



	result=pd.merge(reviews,labels,right_index=True,left_index=True)



	result['Reviews']=result['Reviews'].map(lambda x:x.lower())

	stop=stopwords.words('english')
	result['review_without_stopwords'] = result['Reviews'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))

	# import nltk
	# nltk.download('punkt')


	# # In[19]:


	# nltk.download('average_perceptron_tagger')


	# # In[20]:
	

	def pos(review_without_stopwords):
		return TextBlob(review_without_stopwords).tags
	
	
	os = result.review_without_stopwords.apply(pos)
	os1 = pd.DataFrame(os)

	os1['pos'] = os1['review_without_stopwords'].map(lambda x:" ".join(["/".join(x) for x in x ]) )
	result = result = pd.merge(result, os1,right_index=True,left_index = True)
	
	
	r_train, r_test, l_train, l_test = train_test_split(result['pos'],result['Labels'], test_size=0.2,random_state=13)

	tf_vect = TfidfVectorizer(lowercase = True, use_idf=True, smooth_idf=True, sublinear_tf=False)

	X_train_tf = tf_vect.fit_transform(r_train)
	X_test_tf = tf_vect.transform(r_test)


	# In[29]:


	from sklearn.svm import SVC
	clf=SVC(kernel='linear')

	 
	clf.fit(X_train_tf,l_train)
	

	pred=clf.predict(X_test_tf)
	


	if request.method == "POST":
		comment = request.form["comment"]
		# data = [comment]		
		# vect = tf_vect.transform(data).toarray()
		# my_prediction = clf.predict("hi theres is ")
		# def test_string(data):
		# 	X_test_tf = tf_vect.transform(data)
		# 	y_predict = clf.predict(X_test_tf)
		# 	return render_template('result.html', prediction= y_predict)
		vect= tf_vect.transform([comment]).toarray()
		y_predict = clf.predict(vect)
	return render_template('result.html')	
		# comment = request.form['comment']
		# data = [comment]
	
	# return render_template('result.html', prediction= y_predict)


if __name__ == '__main__':
	app.run(debug=True)