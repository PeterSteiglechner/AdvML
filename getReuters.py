import numpy as np 
import matplotlib.pyplot as plt 
import scipy
from scipy import special
import pandas as pd
import nltk


from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.corpus import reuters


# If you run this cell for the first time, uncomment the following line to download the corpus
nltk.download('reuters')
#reuters.fileids()


training_ids = []
test_ids = []
for name in reuters.fileids():
    if "training" in name:
        training_ids.append(name)
    else:
        test_ids.append(name)
print("M_max_reuters = ", len(training_ids) )


M_reuters=100

training=[]  
for i in range(M_reuters):
    #file=str(i+1)
    file = training_ids[i] # filename
    training.append(reuters.words(file)) #append the document to training

doc_lengths =  [len(training[i]) for i in range(len(training))] # this is N_d (length of each of the documents)
print("We have ", len(training), " Documents with sizes: ", doc_lengths[:100])

#" ".join(training[0])


vocab_list=[] # 

def create_dict(data, test = True):     
    global vocab_list # change the global vocab_list
    D=[]  # dim = (#documents, #N_reduced_d),    where N_reduced_d is the Nr of words that are not filtered / are not unnecessary
    for d in data: # for each document
        # Uncomment 3 lines below to test the function with the string below 
        # test=True
        if test==False: 
            d = " ".join(d)
        d = " ".join(d)
        dclean = remove_unnecessary_words(d)   # list of all the non-unnecessary words of the document
        d_vocabInd = []  # This is the document in terms of the indices of the words in the vocab_list (example above: [0,1])
        d_words=[]
        for w in dclean:  # loop through these words
            w = w.lower() 
            if w in vocab_list: 
                pass # word is already in vocab_list
            else:
                #add word to vocab_list
                vocab_list.append(w)
            
            # Store the word as the document
            #v = vocab_list.index(w) #index v   #OLD
            #d_vocabInd.append(v)   #OLD
            d_words.append(w)
            
        #D.append(d_vocabInd)   # Add document to corpus
        D.append(d_words)   # Add document to corpus
    return D


def remove_unnecessary_words(document):
    words = word_tokenize(document)
    wordsFiltered = []
    for w in words:
        if conditions(w):  # all the conditions what define an unnecessary word
            # word is not unnecessary --> add to cleaned document
            wordsFiltered.append(w)
            
    return wordsFiltered


def conditions(word):
    '''Return True if all conditions are fulfilled'''
    Cond=True
    # Word is no stopword
    Cond *= (word not in stopwords.words('english'))
    # Word is no special sign (like ".")
    Cond *= (word not in ['.', ',', ';', '-', '+',                              '?', '!', '=', '(', ')',                             '/', '&', '$', '€'])
    # Word is no float ("4.2")
    if ("." in word) or ("," in word):
        if word[0].isdecimal():
            Cond=False
    # Word is no digit
    Cond *= (not word.isdigit())
    Cond *= (not word.isdecimal())
    
    return Cond

'''TEST
data_test = ["All work and no play makes 4 jack dull boy.", 
        "All people make mistakes, but 4 5 jack makes no work.",
        "People are dull, jack makes 4.5 mistakes"]

D = create_dict(data_test)
D, vocab_list
'''


vocab_list = [] # init vocab_list
D_reuters = create_dict(training)
#
# len(D_reuters), len(vocab_list)


#lengths = [] # Lengths of the cleaned documents 
#for i in range(len(D_reuters)):
#    lengths.append(len(D_reuters[i]))
#p = plt.hist(lengths, bins=40)
#p = plt.hist(doc_lengths, bins=40, alpha=0.5) # document lengths (uncleaned)##
#
#print("Mean Document length (incl Stopwords etc.", np.mean(doc_lengths))
#print("Mean Lengths of final document (list of words):", np.mean(lengths))
#print("Vocab Size ", len(vocab_list))
# 
#plt.xlabel("Length of a document")
#plt.ylabel("count")
#plt.title("Reuters")


# In[18]:





# In[ ]:




