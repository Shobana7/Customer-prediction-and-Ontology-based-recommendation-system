#!/usr/bin/env python
# coding: utf-8

# In[3]:


import json
import io
import nltk
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize 
from nltk.tokenize import RegexpTokenizer
from collections import Counter
import pandas as pd

import nltk
nltk.download('stopwords')


# In[16]:


#Extracting reviews of products whose rating >=4
#need to run like this seperately for each category

data = pd.read_csv("DataFiles/Videogames")
file = open("DataFiles/EachCategoryTop200.txt","w+")

for x in data.index:
    if(data['overall'][x]== 4 or data['overall'][x]==5):
        json_review =  data['reviewText'][x]
        file.write(str(json_review))
        file.write("\n")
        
file.seek(0)


# In[17]:


#Removing stop words and punctuations
stop_words = set(stopwords.words('english')) 
line = file.read()
tokenizer = RegexpTokenizer(r'\w+')
result = tokenizer.tokenize(line)
for r in result: 
    if not r in stop_words: 
        appendFile = open('DataFiles/EachCategoryTop200.txt','a') 
        appendFile.write(" "+r) 
        appendFile.close() 


# In[18]:


file1 = open('DataFiles/EachCategoryTop200.txt','r')
lines = file1.read()
adjectives = [word for (word, pos) in nltk.pos_tag(nltk.word_tokenize(lines))if (pos == 'JJ' or pos == 'JJR' or pos == 'JJS')]
file2 = open('DataFiles/EachCategoryTop201.txt','w+')
for element in adjectives:
    file2.write(element)
    file2.write('\n')
file2.close()
file1.close()


# In[19]:


file1 = open('DataFiles/EachCategoryTop201.txt','r')
lines = file1.read()
counts = Counter(lines.split())
top_five = counts.most_common(200)
res = list(zip(*top_five)) 
aList = list(res[0])
appendFile = open('DataFiles/Videogames_200.txt','a') 
for element in aList:
    appendFile.write(element)
    appendFile.write('\n')

appendFile.close()


# In[ ]:



  


# In[ ]:




