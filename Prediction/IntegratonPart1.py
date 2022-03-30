#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import warnings
warnings.filterwarnings("ignore")


# In[1]:


def create_reviews_data():
    
    import json
    from pandas.io.json import json_normalize
    import pandas as pd

    reviewerData = pd.read_csv('AmazonReviews')  #using the created reviews json file created      #all data in the file is under the 'test' tag
    reviewerDF = reviewerData[['asin', 'reviewerID', 'reviewerName', 'main_cat', 'price']].copy()
    reviewerDF.rename(columns = {'asin':'ProductIDs', 'reviewerID':'ID', 'reviewerName' : 'Name', 'main_cat' : 'AllCategories', 'price' : 'Price'}, inplace = True)
    reviewerDF['Price'] = reviewerDF['Price'].str[1:].astype(float)  #need to remove the $ which is part of 'Price'
    
    return reviewerDF


# In[2]:


# find the average, min and max prices for each reviewer. Is needed later for Neighbour KNN

def create_price_features(reviewerDF):
    
    import numpy as np
    gb = reviewerDF.groupby(['Name', 'ID'], as_index = False)  
    df2 = gb.agg({'Price' : [np.mean, np.min, np.max]})
    df2.columns = ['Name', 'ID', 'AvgPrice', 'MinPrice', 'MaxPrice']
    
    return df2


# In[3]:


# grouping together all the products, categories of products purchased. Needed later to create a comprehensive reviewers database for later. And for neighbourKNN

def group_products_categories(reviewerDF):
    
    import pandas as pd
    
    #getting a list of all items bought by user
    df3 = pd.DataFrame()
    df3 = reviewerDF.groupby(['ID'])['ProductIDs'].apply(list).reset_index(name='ProductIDs')

    #getting a list of all categories bought by user
    df4 = pd.DataFrame()
    df4 = reviewerDF.groupby(['ID'])['AllCategories'].apply(list).reset_index(name='AllCategories')

    return df3, df4


# In[4]:


#combine the previously created dataframes to create our final reviewersDF

def combine_reviewer_df(df2, df3,df4):
    
    import pandas as pd
    
    middledf = pd.merge(df2, df3, on="ID")
    reviewerDF = pd.merge(middledf,df4,on = 'ID')

    return reviewerDF


# In[5]:


##all function calls related to creating reviewer data
#unlike in the case of product related data, some features here are a list. Is creating problem to save and reload. Hmece building this dataframe here

reviewerDF = create_reviews_data()
print("The data has been loaded from the json file")
#print(reviewerDF)

#reviewerDF = append_data(reviewerDF)
#print("The dummy data has been appended")
#print(reviewerDF)

df2 = create_price_features(reviewerDF)
print("A new dataframe that has all the prices has been created")
#print(df2)

df3, df4 = group_products_categories(reviewerDF)
print("Seperate dataframes having collection of products and collection of categories have been created")
#print("The products related frame is")
#print(df3)
#print("The categories related frame is")
#print(df4)

reviewerDF = combine_reviewer_df(df2, df3,df4)
print("The final reviewerDF to be used has been created")
#print(reviewerDF)


# In[6]:


#creating the product dataframe from previously created ProductDF. Similar procedure has to be done later for reviewer data also

def create_product_data():
    
    import pandas as pd
    import json
    from pandas.io.json import json_normalize
    
    productDF = pd.read_csv('ProductDF') #dataframe loaded from already created csv file
    productDF['Features'] = productDF['Features'].apply(lambda x: x[1:-1]) #remove the opening and closing brackets around the ProductFeatures data
    #productDF['TopFiveWords'] = productDF['TopFiveWords'].apply(lambda x: x[2:-2]) #remove the two opening and closing brackets around the TopFiveWords data
    
    return productDF


# In[7]:


#all function calls relating to product data

productDF = create_product_data()
print("The final productDF to be used has been created")
#print(productDF)


# In[8]:


def setup_ontology():
        
    #loading our ontology
    onto = get_ontology("CompletedOntology.owl") #can be found at the below address
    onto_path.append("C:\MyProgramFiles\Anaconda3\Lib\owlready2")
    onto.load()
    
    return onto


# In[9]:


#Sending all Product information

def send_product_data(productDF, onto):
    
    #converting the data types before sending 
    productDF['Price'] = productDF['Price'].astype(str)
    productDF['Rating'] = productDF['Rating'].astype(str)

    for ind in productDF.index: 
        testProduct = onto.Product(productDF['ID'][ind], 
                             ProductCategory = [productDF['Category'][ind]],
                             ProductFeatures = [productDF['Features'][ind]],
                             ProductID = [productDF['ID'][ind]],
                             ProductPrice  = [productDF['Price'][ind]],
                             ProductRating = [productDF['Rating'][ind]],
                             ProductName = [productDF['ProductName'][ind]])
    onto.save()
    
    return productDF, onto
    


# In[10]:


#Sending all Reviewer information

def send_reviewer_data(reviewerDF, onto):
    
    import pandas as pd
    
    #converting the data types before sending - this line was not needed in the function based code. is present in code that was written in a sequence
    reviewerDF['AvgPrice'] = reviewerDF['AvgPrice'].astype(str)
    reviewerDF['MinPrice'] = reviewerDF['MinPrice'].astype(str)
    reviewerDF['MaxPrice'] = reviewerDF['MaxPrice'].astype(str)
    
    Recommendation = onto
    
    for ind in reviewerDF.index: 
        testPerson = onto.Person(reviewerDF['ID'][ind], 
                             ReviewerName = [reviewerDF['Name'][ind]],
                             AveragePrice = [reviewerDF['AvgPrice'][ind]],
                             ReviewerID = [reviewerDF['ID'][ind]])
    
        for product in reviewerDF['ProductIDs'][ind]:
            ontoProd = onto.Product(product)
            testPerson.hasBoughtProducts.append(ontoProd)
    
   
    onto.save()
    
    return reviewerDF, onto
    


# In[11]:


#all function calls to do with creating our ontology

from owlready2 import *

onto = setup_ontology()
print("The ontology has been created")
#print(onto)

productDF, onto = send_product_data(productDF, onto)
print("The product related data has been sent")

reviewerDF, onto = send_reviewer_data(reviewerDF, onto)
print("The reviewer related data has been sent")


# In[12]:


#saving any changes made to the dataframe so that they can be used in the next module

def save_dataframes(productDF, reviewerDF):
    
    import pandas as pd
    import csv
    
    productDF.to_csv('RevisedProductDF')
    reviewerDF.to_csv('RevisedReviewerDF')
    


# In[13]:


#all function calls to do with saving dataframes

save_dataframes(productDF, reviewerDF)
print("The dataframes have been saved")

