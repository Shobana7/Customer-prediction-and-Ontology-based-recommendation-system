#!/usr/bin/env python
# coding: utf-8

# In[1]:


#importing necessary libs
import json
import io
import nltk
from pandas.io.json import json_normalize
import pandas as pd


# In[2]:


productData = json.load(open("All_Product.json"))
productDF = json_normalize(productData['test'])
productDF.rename(columns = {'asin':'ID',
                            'price' : 'Price',
                            'overall' : 'Rating',
                            'category' : 'Category',
                            'features' : 'ProductFeatures',
                             'title': 'ProductName'}, inplace = True)
#print(productDF)


# In[3]:


productDF.to_csv('ProductDF')


# In[ ]:




