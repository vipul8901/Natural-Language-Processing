#train_path = "../resource/lib/publicdata/aclImdb/train/" # use terminal to ls files under this directory
#test_path = "../resource/lib/publicdata/imdb_te.csv" # test data for grade evaluation

train_path = "./aclImdb/train/" # replace this path before submitting
test_path = "./"

#name="imdb_tr.csv"
import os
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import SGDClassifier
import re

imdb_tr=pd.DataFrame()

"""tst=[]
st="abcdef"
tst=tst+[st]
print(type(st))"""

"""txt,txtfinal="",""
txtsplit,txtfiltered=[],[]"""
#global vocab,vectorizer
#vocab=[]

def noiseremoval(text):
    # remove html markup
    text=re.sub("(<.*?>)","",text)
    
    #remove non-ascii and digits
    text=re.sub("(\\W|\\d)"," ",text)
    
    #remove whitespace
    text=text.strip()
    return text

def imdb_data_preprocess(inpath, outpath="./", name="imdb_tr.csv", mix=False):
    
    file_list=os.listdir(inpath)
    filetxt=[]
    row_num=[]
    polarity=[]
    pol=0
    sz=0
    #global vocab,vectorizer
    #print(inpath[-4:-1])
    if inpath[-4:-1]=="pos":
        pol=1
    else:
        sz=len(imdb_trp)
    for cntr,i in enumerate(file_list):
        """print(inpath[-4:-1])
        print(cntr)
        print(i)"""
        #if cntr >=2:
        """if (cntr >=1) or (inpath[-4:-1]!="pos"):
            break"""
        #with open(inpath+i,"r") as fr: #error in file 10327_7.txt throwing codec error. Hence used next line of code to ignore such errors
        #with open(inpath+i,"r",encoding="utf8") as fr: #This worked absolutely fine too
        with open(inpath+i,"r",encoding="ISO-8859-1") as fr:
        #with open(inpath+i,"r",errors='ignore') as fr: # This one worked as well but the file in question had issue. in particular ' was read as a weird code
            txt=str(fr.read())
            txtsplit=txt.split()
            txtfiltered=[noiseremoval(word.lower()) for word in txtsplit if word.lower() not in stopwords]
            txtfinal=" ".join(txtfiltered)
            #vocab=vocab+[word for word in txtsplit if word.lower() not in stopwords+vocab]
            #vocab=vocab+txtfiltered
            
            """print(txtsplit)
            for word in txtsplit:
                print(word.lower())"""
                #if word.lower() not in stopwords
            #print(type(txt))
            #filetxt=filetxt+[txt]
            filetxt=filetxt+[txtfinal]
            #print(fr.read())
        polarity=polarity+[pol]
        rwn=cntr+1+sz
        row_num=row_num+[rwn]
    #print(len(filetxt))
    
    """vectorizer = CountVectorizer()
    vocab=vectorizer.fit_transform(filetxt)"""
    
    inp={"row_number":row_num,"text":filetxt,"polarity":polarity}
    #imdb_tr=pd.DataFrame(data=[row_num,filetxt,polarity],index=range(0,len(file_list)),columns=["row_number","text","polarity"])
    imdb_t=pd.DataFrame(inp)
    return imdb_t
    #print(filetxt)
            
    #print(filetxt)
    '''Implement this module to extract
	and combine text files under train_path directory into 
    imdb_tr.csv. Each text file in train_path should be stored 
    as a row in imdb_tr.csv. And imdb_tr.csv should have two 
    columns, "text" and label'''
    pass

"""def tokenize(filetxt,gram=(1,1),voc=None):
    
    if voc==None:
        vectorizer = CountVectorizer(ngram_range=gram,max_df=.2)
        vocab=vectorizer.fit_transform(filetxt)
        vocablry=vectorizer.get_feature_names()
        #docterm_f=vocab.toarray()
        #return vocablry,docterm_f
        return vocablry
    else:
        vectorizer = CountVectorizer(ngram_range=gram,vocabulary=voc)
        vocab=vectorizer.transform(filetxt)
        #vocablry=vectorizer.get_feature_names()
        docterm_f=vocab.toarray()
        return docterm_f"""

def vocab_gen(filetxt,gram=(1,1)):
    vectorizer = CountVectorizer(ngram_range=gram,lowercase=True)
    vocab=vectorizer.fit(filetxt)
    return vocab

def idf_gen(sparse_mat):
    idfvec = TfidfTransformer()
    idf=idfvec.fit(sparse_mat)
    return idf
    
def model_fit(x,y):
    sgc=SGDClassifier(penalty="l1")
    model=sgc.fit(x,y)
    return model
    
def accuracy(pred):
    return sum(abs(imdb_tr.polarity-pred))

def accuracy_p(pred):
    return 1-(sum(abs(imdb_tr.polarity-pred))/len(imdb_tr.polarity))

def writeout(name,my_list):
    with open(name, 'w') as f:
        for item in my_list:
            f.write("%s\n" % item)
    
    
if __name__ == "__main__":
    #pass
    f = open("stopwords.en.txt", "r")
    stopwords=[]
    for x in f:
      #print(x)
      stopwords+=[x.strip().lower()]
    imdb_trp=imdb_data_preprocess(train_path+"pos/")
    imdb_trn=imdb_data_preprocess(train_path+"neg/")
    imdb_tr=imdb_trp.append(imdb_trn,ignore_index=True)
    
    
    imdb_te=pd.read_csv(test_path+"imdb_te.csv",encoding = "ISO-8859-1")
    
    #vocablry_uni,docterm_f_uni=tokenize(imdb_tr.text)
    #vocablry_uni=tokenize(imdb_tr.text)
    #vocablry_bi,docterm_f_bi=tokenize(imdb_tr.text,gram=(2,2))
    #vocablry_bi=tokenize(imdb_tr.text,gram=(2,2))
    
    unigram=vocab_gen(imdb_tr.text)
    bigram=vocab_gen(imdb_tr.text,gram=(1,2))
      
    unigram_dtf_train=unigram.transform(imdb_tr.text)
    bigram_dtf_train=bigram.transform(imdb_tr.text)
    
    unigram_idf=idf_gen(unigram_dtf_train)
    bigram_idf=idf_gen(bigram_dtf_train)
    
    unigram_idf_train=unigram_idf.transform(unigram_dtf_train)
    bigram_idf_train=bigram_idf.transform(bigram_dtf_train)
    
    
    unigram_dtf_tst=unigram.transform(imdb_te.text)
    bigram_dtf_tst=bigram.transform(imdb_te.text)
    
    unigram_idf_tst=unigram_idf.transform(unigram_dtf_tst)
    bigram_idf_tst=bigram_idf.transform(bigram_dtf_tst)
    
    """commenting print
    print(unigram_dtf_train.shape)
    print(bigram_dtf_train.shape)
    print(unigram_dtf_tst.shape)
    print(bigram_dtf_tst.shape)
    
    print(unigram_idf_train.shape)
    print(bigram_idf_train.shape)
    print(unigram_idf_tst.shape)
    print(bigram_idf_tst.shape)"""
    
    unigram_model=model_fit(unigram_dtf_train,imdb_tr.polarity)
    bigram_model=model_fit(bigram_dtf_train,imdb_tr.polarity)
    unigram_model_idf=model_fit(unigram_idf_train,imdb_tr.polarity)
    bigram_model_idf=model_fit(bigram_idf_train,imdb_tr.polarity)
    
    
    unigram_predict=unigram_model.predict(unigram_dtf_tst)
    bigram_predict=bigram_model.predict(bigram_dtf_tst)
    unigram_predict_idf=unigram_model_idf.predict(unigram_idf_tst)
    bigram_predict_idf=bigram_model_idf.predict(bigram_idf_tst)
    
    """commenting print
    print(unigram_predict.sum())
    #print(accuracy(unigram_predict))
    print(bigram_predict.sum())
    #print(accuracy(bigram_predict))
    print(unigram_predict_idf.sum())
    #print(accuracy(unigram_predict_idf))
    print(bigram_predict_idf.sum())
    #print(accuracy(bigram_predict_idf))"""
    
    writeout("unigram.output.txt",unigram_predict)
    writeout("bigram.output.txt",bigram_predict)
    writeout("unigramtfidf.output.txt",unigram_predict_idf)
    writeout("bigramtfidf.output.txt",bigram_predict_idf)
    
    #Predicting on Training data
    
    tunigram_predict=unigram_model.predict(unigram_dtf_train)
    tbigram_predict=bigram_model.predict(bigram_dtf_train)
    tunigram_predict_idf=unigram_model_idf.predict(unigram_idf_train)
    tbigram_predict_idf=bigram_model_idf.predict(bigram_idf_train)
    
    
    """commenting print
    print(tunigram_predict.sum())
    print(accuracy(tunigram_predict), "and ", accuracy_p(tunigram_predict))
    print(tbigram_predict.sum())
    print(accuracy(tbigram_predict), "and ", accuracy_p(tbigram_predict))
    print(tunigram_predict_idf.sum())
    print(accuracy(tunigram_predict_idf), "and ", accuracy_p(tunigram_predict_idf))
    print(tbigram_predict_idf.sum())
    print(accuracy(tbigram_predict_idf), "and ", accuracy_p(tbigram_predict_idf))
    """
    #print(accuracy(np.zeros(25000)))
    """vectorizer = CountVectorizer(stop_words='english')
    vocab=vectorizer.fit_transform(imdb_tr.text)
    tmp=vectorizer.get_feature_names()
    tmp2=vocab.toarray()"""
    
    
    """if "it" in stopwords:
        print("Yes")"""
    
    '''train a SGD classifier using unigram representation,
    predict sentiments on imdb_te.csv, and write output to
    unigram.output.txt'''
  	
    '''train a SGD classifier using bigram representation,
    predict sentiments on imdb_te.csv, and write output to
    bigram.output.txt'''
    
    '''train a SGD classifier using unigram representation
     with tf-idf, predict sentiments on imdb_te.csv, and write 
     output to unigramtfidf.output.txt''''''train a SGD classifier using bigram representation
     with tf-idf, predict sentiments on imdb_te.csv, and write 
     output to bigramtfidf.output.txt'''
     