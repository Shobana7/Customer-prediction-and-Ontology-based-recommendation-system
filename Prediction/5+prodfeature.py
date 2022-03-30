#!/usr/bin/env python
# coding: utf-8

# In[1]:


import json
import io
import nltk
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize 
from nltk.tokenize import RegexpTokenizer
from collections import Counter
import pandas as pd


# In[67]:


#Extracting reviews of products whose rating >=4
data = pd.read_csv("DataFiles/Videogames")
file = open("DataFiles/prodfeature.json","w+")


data1 = {}
data1['test'] = []

for x in data.index:
            json_review = str(data['reviewText'][x])
        #print(json_review)
            adjectives = [word for (word, pos) in nltk.pos_tag(nltk.word_tokenize(json_review))if (pos == 'JJ' or pos == 'JJR' or pos == 'JJS')]
            counts = Counter(adjectives)
            top_five = counts.most_common(5)
            res = list(zip(*top_five)) 
            aList = list(res[0:1])
            data1['test'].append( {
                "asin": data['asin'][x],
                "top5words" : [element for element in aList],
                "overall": data['overall'][x],
                "price": data['price'][x],
                "category": data['main_cat'][x],
                "title":data['title'][x]
            })

            
json.dump(data1, file,indent=0)

file.seek(0)


# In[68]:


data2 = json.load(open("DataFiles/prodfeature.json"))
file2 =  open('DataFiles/Videogames_200.txt','r') 
file3 = open('DataFiles/IntermediateProducts.json','w+')


# In[69]:


id = []
for x in range(0,4000):
     if "asin" in data2['test'][x]:
            id.append(data2['test'][x]['asin'])
        

#print(id)
k = len(id)
#print(k)
file.seek(0)


# In[70]:


rows,cols = (k,3)

arr = [[0 for i in range(cols)] for j in range(rows)]
#[prod_id    withinthernge?  raitng    if top5words are in top200]
j=0 

data1 = {}
data1['test'] = []

contents = file2.read()
while(j<k):
                arr[j][0] = id[j]
                arr[j][1] = data2['test'][j]['overall']
                if( data2['test'][j]['top5words'] is not []):
                    y=len(data2['test'][j]['top5words'])
                    if y:
                         for element in data2['test'][j]['top5words']:
                                for word in element:
                                     if(word in contents):
                                        arr[j][2]+=1
                            
                j+=1
    
#for row in arr: 
#    print(row) 
    
file.seek(0)
for x in range(0,4000):
    if(data2['test'][x]['asin'] == arr[x][0]):
        data1['test'].append( {
                "asin": data2['test'][x]['asin'],
                "top5words" : data2['test'][x]['top5words'],
                "overall": data2['test'][x]['overall'],
                "price": data2['test'][x]['price'],
                "category": data2['test'][x]['category'],
                "title": data2['test'][x]['title'],
                "features" : arr[x][1:4],
            })

json.dump(data1, file3,indent=0)


# In[71]:


from pandas.io.json import json_normalize
df = pd.DataFrame()
df = json_normalize(data1['test'])
#df


# In[72]:


import numpy as np
df1 = df.drop(['top5words','category'],axis=1)
df1['price'] = df1['price'].str[1:].astype(float)
df1['overall'] = df1['overall'].astype(float)


# In[73]:


df2 = df1.groupby(['asin','title'], as_index=False).agg({'overall':'mean', 'price':'mean'})
#df2


# In[74]:


df3 = pd.DataFrame()
df3 = df.drop(['top5words','category','overall','price','title'],axis=1)
#df3


# In[75]:


import numpy
for ind in df3.index:
    df3['features'][ind]=numpy.array(df3['features'][ind])  


# In[76]:


df4 = pd.DataFrame()
df4 = df3.groupby('asin')['features'].apply(numpy.mean).apply(list).reset_index(name='Features')


# In[77]:


df5 = pd.merge(df4,df2, on= 'asin')


# In[78]:


df5['category']= 'Videogames'
#df5


# In[79]:


df5.to_json(r'DataFiles/Videogames_Product.json',orient='records')


# In[ ]:




