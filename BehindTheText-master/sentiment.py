import nltk 
import numpy as np 
from sklearn.utils import shuffle 
from nltk.corpus import stopwords 
from nltk.stem.porter import PorterStemmer 
from sklearn.linear_model import LogisticRegression 
from bs4 import BeautifulSoup 
 
 
ps = PorterStemmer() 
 
positive_reviews = BeautifulSoup(open('positive.review').read()) 
positive_reviews = positive_reviews.findAll('review_text') 
 
negative_reviews = BeautifulSoup(open('negative.review').read()) 
negative_reviews = negative_reviews.findAll('review_text') 
diff = len(positive_reviews) - len(negative_reviews) 
idxs = np.random.choice(len(negative_reviews), size=diff) 
extra = [negative_reviews[i] for i in idxs] 
negative_reviews += extra 
 
def my_tokenizer(s): 
    s = s.lower()     
    tokens = nltk.tokenize.word_tokenize(s)      
    tokens = [t for t in tokens if len(t) > 2]     
    tokens = [ps.stem(t) for t in tokens]  
    tokens = [t for t in tokens if t not in set(stopwords.words('english'))]      
    return tokens 
 
word_index_map = {} 
current_index = 0 
orig_reviews = [] 
 
positive_tokenized = [] 
negative_tokenized = [] 
for review in positive_reviews:     
    orig_reviews.append(review.text)     
    tokens = my_tokenizer(review.text)     
    positive_tokenized.append(tokens)     
    for token in tokens:         
        if token not in word_index_map:             
            word_index_map[token] = current_index 
            current_index += 1 
 
for review in negative_reviews:    
    orig_reviews.append(review.text)     
    tokens = my_tokenizer(review.text)     
    negative_tokenized.append(tokens)     
    for token in tokens:         
        if token not in word_index_map:            
            word_index_map[token] = current_index 
            current_index += 1 
def tokens_to_vector(tokens, label): 
    x = np.zeros(len(word_index_map) + 1)     
    for t in tokens:         
        i = word_index_map[t] 
        x[i] += 1    
        x = x / x.sum()     
        x[-1] = label 
    return x 
 
N = len(positive_tokenized) + len(negative_tokenized) 
 
data = np.zeros((N, len(word_index_map) + 1)) 
i = 0 
for tokens in positive_tokenized:     
    xy = tokens_to_vector(tokens, 1) 
    data[i,:] = xy     
    i += 1 
 
for tokens in negative_tokenized:     
    xy = tokens_to_vector(tokens, 0) 
    data[i,:] = xy 
    i += 1 
 
orig_reviews, data = shuffle(orig_reviews, data) 
 
x	= data[:,:-1] 
y	= data[:,-1] 
 
Xtrain = x[:-100,] 
Ytrain = y[:-100,] 
 
model = LogisticRegression()
model.fit(Xtrain, Ytrain) 
print("Accuracy:", model.score(Xtrain, Ytrain)) 
 
preds = model.predict(x) 
p = model.predict_proba(x)[:,1] 
prob = p.sum()/len(p) 
if(prob>0.6):     
    print("Review is neutral") 
elif(0.4<prob<0.6): 
    print("Review is positive") 
else:     
   print("Review is negative")
