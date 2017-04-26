#import regex
import re
import csv
import xlrd
import nltk
import svm
from svmutil import *

#start process_tweet
def processTweet(tweet):
    # process the tweets

    #Convert to lower case
    tweet = tweet.lower()
    #Convert www.* or https?://* to URL
    tweet = re.sub('(www\.[^\s]+)|(https?://[^\s]+)|(http?://[^\s]+)|([^\s]+.com)|([^\s]+.net)|([^\s]+.in)|([^\s]+@[^\s]+)','URL',tweet)
    #Convert @username to AT_USER
    tweet = re.sub('@[^\s]+','AT_USER',tweet)
    #Remove additional white spaces
    tweet = re.sub('[\s]+', ' ', tweet)
    #Replace #word with word
    tweet = re.sub(r'#([^\s]+)', r'\1', tweet)
    #trim
    tweet = tweet.strip('\'"')
    #emoticon removal
    myre = re.compile(u'['u'\U0001F300-\U0001F64F'u'\U0001F680-\U0001F6FF'u'\u2600-\u26FF\u2700-\u27BF]+', re.UNICODE)
    tweet=myre.sub('', tweet)

    return tweet
#end


#start replaceTwoOrMore
def replaceTwoOrMore(s):
    #look for 2 or more repetitions of character and replace with the character itself
    pattern = re.compile(r"(.)\1{1,}", re.DOTALL)
    return pattern.sub(r"\1\1", s)
#end

#start getStopWordList
def getStopWordList(stopWordListFileName):
    #read the stopwords file and build a list
    stopWords = []
    stopWords.append('AT_USER')
    stopWords.append('URL')

    fp = open(stopWordListFileName, 'r')
    line = fp.readline()
    while line:
        word = line.strip()
        stopWords.append(word)
        line = fp.readline()
    fp.close()
    return stopWords
#end

#start getfeatureVector
def getFeatureVector(tweet):
    featureVector = []
    #split tweet into words
    words = tweet.split()
    for w in words:
        #replace two or more with two occurrences
        w = replaceTwoOrMore(w)
        #strip punctuation
        w = w.strip('\'"?,.')
        #check if the word stats with an alphabet
        val = re.search(r"^[a-zA-Z][a-zA-Z0-9]*$", w)
        #ignore if it is a stop word
        if(w in stopWords or val is None):
            continue
        else:
            featureVector.append(w.lower())
    return featureVector
#end


def extract_features(tweet):
    tweet_words = set(tweet)
    features = {}
    for word in featureList:
        features['contains(%s)' % word] = (word in tweet_words)
    return features


def getSVMFeatureVectorandLabels(tweets, featureList):
    sortedFeatures = sorted(featureList)
    map1 = {}
    feature_vector = []
    labels = []
    val=[]
    for t in tweets:
        label = 1
        map1 = {}
        #Initialize empty map
        for w in sortedFeatures:
            map1[w] = 0

        tweet_words = t[0]
        tweet_opinion=t[1]
        #print (tweet_opinion)
        #Fill the map
        for word in tweet_words:
            #process the word (remove repetitions and punctuations)
            word = replaceTwoOrMore(word)
            word = word.strip('\'"?,.')
            #set map[word] to 1 if word exists
            if word in map1:
                map1[word] = 1
        #end for loop

         
        val = list(map1.values())
        feature_vector.append(val)
        if(tweet_opinion == 1):
            label = 1
        elif(tweet_opinion == 0):
            label = 0
        #elif(tweet_opinion == 'neutral'):
        #    label = 2
        #print (label)
        label=float(label)
        labels.append(label)
    #return the list of feature_vector and labels
    return {'label': labels,'feature_vector' : feature_vector }
#end

#initialize stopWords
stopWords = []
st = open(r'F:\sem7\project\svm_impl\stopWord_list.txt', 'r')
stopWords = getStopWordList(r'F:\sem7\project\svm_impl\stopWord_list.txt')


#Read the tweets one by one and process it
wb = xlrd.open_workbook(r'F:\sem7\project\svm_impl\datasets\lenovoK3\Training5.xls')
sheet = wb.sheet_by_index(0)
max_row=sheet.nrows
tweets=[]
featureList = []
for i in range(max_row):
    tweet=sheet.cell(i,0).value
    sentiment=sheet.cell(i,1).value
    processedTweet = processTweet(tweet)
    featureVector = getFeatureVector(processedTweet)
    featureList.extend(featureVector)
    tweets.append((featureVector, sentiment));
    #print (featureVector)
    #print (sentiment)
    #print ("XXXXXXXXXXX")

featureList = list(set(featureList))
#print (len(featureList))
training_set = nltk.classify.util.apply_features(extract_features, tweets)
#print (featureList)
#print (training_set)


wb1 = xlrd.open_workbook(r'F:\sem7\project\svm_impl\datasets\lenovoK3\Test5.xls')
sheet1 = wb1.sheet_by_index(0)
max_row1=sheet1.nrows
tweets1=[]
featureList1 = []
for i in range(max_row1):
    tweet1=sheet1.cell(i,0).value
    sentiment1=sheet1.cell(i,1).value
    processedTweet1 = processTweet(tweet1)
    featureVector1 = getFeatureVector(processedTweet1)
    featureList1.extend(featureVector1)
    tweets1.append((featureVector1, sentiment1));
    #print (featureVector)
    #print (sentiment)
    #print ("XXXXXXXXXXX")




#Train the classifier

result = getSVMFeatureVectorandLabels(tweets, featureList)
#print (result['label'])
problem = svm_problem(result['label'],result['feature_vector'])
#'-q' option suppress console output
param = svm_parameter('-q')
param.C = 10
param.kernel_type = LINEAR
classifier = svm_train(problem, param)
#svm_save_model(classifierDumpFile, classifier)

#Test the classifier
test_v = getSVMFeatureVectorandLabels(tweets1, featureList)
actual_result=test_v['label']
test_feature_vector=test_v['feature_vector']
#p_labels contains the final labeling result
p_labels, p_accs, p_vals = svm_predict([0] * len(test_feature_vector),test_feature_vector, classifier)

#print (p_labels)
#print (actual_result)
#print (p_accs)
#print (p_vals)
#c=set(p_labels)&set(actual_result)
#cnt=len(c)
cnt=0
for i in range(len(actual_result)): 
    if p_labels[i]==actual_result[i]:
                cnt=cnt+1

cnt_p=cnt/(len(actual_result))*100        
print ("SVM Accuracy")
print (cnt_p)

# NAIVE BAYES
#training

NBClassifier = nltk.NaiveBayesClassifier.train(training_set)

cnt1=0
for i in range(max_row1):
    tweet1=sheet1.cell(i,0).value
    sentiment1=sheet1.cell(i,1).value
    processedTweet=processTweet(tweet1)
    nb_ans=NBClassifier.classify(extract_features(getFeatureVector(processedTweet)))
    #mec_ans=MaxEntClassifier.classify(extract_features(getFeatureVector(processedTweet)))
    if nb_ans==sentiment1:
        cnt1=cnt1+1
   
cnt_p1=cnt1/(max_row1)*100        
print ("NAIVE BAYES Accuracy")
print (cnt_p1)


#MAX ENTROPY
cnt2=0
tpl=[]
training_set1=[]
for i in range(len(training_set)):
    list11=training_set[i]
    #tpl.append(list11[0])
    if training_set[i][1]==1.0:
        #print ("This is working")
        tpl.insert(i,(training_set[i][0],"positive"))
    if training_set[i][1]==0.0:
        #print ("this is not")
        tpl.insert(i,(training_set[i][0], "negative"))
    #print (len(tpl))
    #tpl=tuple(tpl)5
    #print (tpl)
    #training_set1.append(tpl)
    #print (training_set1)
    #break
    #tpl=list(tpl)
    #tpl.clear()
    #print (len(tpl))

MaxEntClassifier= nltk.classify.maxent.MaxentClassifier.train(tpl,'GIS', trace=3 , encoding=None, labels=None, gaussian_prior_sigma=0, max_iter = 10)
for i in range(max_row1):
    tweet1=sheet1.cell(i,0).value
    sentiment1=sheet1.cell(i,1).value
    if sentiment1==0.0:
        sent1="negative"
    if sentiment1==1.0:
        sent1="positive"
    processedTweet=processTweet(tweet1)
    mec_ans=MaxEntClassifier.classify(extract_features(getFeatureVector(processedTweet)))
    if mec_ans==sent1:
        cnt2=cnt2+1
cnt_p2=cnt2/(max_row1)*100        
print ("MAXIMUM ENTROPY Accuracy")
print (cnt_p2)
