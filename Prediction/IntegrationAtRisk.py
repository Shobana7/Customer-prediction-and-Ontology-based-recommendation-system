#!/usr/bin/env python
# coding: utf-8

# The webpage should call 'run_recommendation_for_input(custId, prodID)'

# In[1]:


import warnings
warnings.filterwarnings("ignore")


# In[2]:


def start_timer():
    
    import time 

    start_time = time.time()
    return start_time

def stop_timer(start_time):
    
    import time 

    # convert this into milliseconds for statsd
    resp_time = (time.time() - start_time)*1000
    return resp_time


# In[3]:


def load_datasets():
    
    import pandas as pd
    import ast

    productDF = pd.read_csv('RevisedProductDF')
    reviewerDF = pd.read_csv('RevisedReviewerDF')
    
    reviewerDF['ProductIDs'] = reviewerDF['ProductIDs'].apply(ast.literal_eval)  #these two columns get stored as a string instead list. Need to convert them. 
    reviewerDF['AllCategories'] = reviewerDF['AllCategories'].apply(ast.literal_eval)
    
    reviewerDF['MinPrice'] = reviewerDF['MinPrice'].apply(str) #during saving and loading, string got converted to float
    reviewerDF['MaxPrice'] = reviewerDF['MaxPrice'].apply(str)
    reviewerDF['AvgPrice'] = reviewerDF['AvgPrice'].apply(str)
    
    return productDF, reviewerDF


# In[4]:


def load_ontology():
    
    import owlready2 as ol
    
    #loading our ontology
    onto = ol.get_ontology("IntegratedOntology.owl") #can be found at the below address
    ol.onto_path.append("/home/woof-woof/Desktop/Prediction")
    onto.load()
    print(onto)
    return onto


# In[5]:


##will take the input during testing. For now, assume this is the input

def user_info(reviewerDF, personOfInterest, productOfInterest):

    #active_user = 'A1055QXUA6BOEL' 
    active_user = personOfInterest
    #active_product ='B00000I1C0'  #'0005092663' 
    active_product = productOfInterest
    try:
        activeIndex = reviewerDF[reviewerDF['ID']==active_user].index.values.astype(int)[0]
    except:
        return 0,0,0

    return active_user, active_product, activeIndex   


# In[6]:


##needed to update the active user from 'Person' to 'User'

def create_neighbour_list(reviewerDF, activeIndex, active_product):
    
    ActiveUserProductList = active_product
    
    NeighboursList = []
    for ind in reviewerDF.index :
        if(ActiveUserProductList in reviewerDF['ProductIDs'][ind]) :
            if(ind!=activeIndex) : 
                newProductSet = set(reviewerDF['ProductIDs'][ind]).difference(ActiveUserProductList)
                if(len(newProductSet)!=0) :
                    NeighboursList.append(reviewerDF['ID'][ind])
                    
    if not NeighboursList:
        return 0
    
    return NeighboursList


# In[7]:


#needed to update the active user from 'Person' to 'User'

def pick_min_max_price(reviewerDF, activeIndex):
    
    minPriceActiveUser = reviewerDF['MinPrice'][activeIndex]
    maxPriceActiveUser = reviewerDF['MaxPrice'][activeIndex]

    return minPriceActiveUser, maxPriceActiveUser


# In[8]:


#to send data to ontology to update 'Person' to 'User'

def update_in_ontology(onto, active_user, minPriceActiveUser, maxPriceActiveUser, NeighboursList):
    
    onto.Person(active_user, MinPrice = [minPriceActiveUser], MaxPrice = [maxPriceActiveUser])
    onto.save()

    for neighbour in NeighboursList :
        ontoNeighbour = onto.Person(neighbour)
        onto.Person(active_user).hasNeighbours.append(ontoNeighbour)
        
    onto.save()
    
    return onto


# In[9]:


#creating feature for each neighbour

def neighbour_feature_creation(NeighboursList, reviewerDF):
    
    k = len(NeighboursList)
    #have to change number of columns. Will be # of categories + 2
    rows,cols = (k,6)
    neighbourArray = [[0 for i in range(cols)] for j in range(rows)]
    
    j=0 
    while(j<k):
        neighbourArray[j][0] = NeighboursList[j]  #first element is Person ID
        neighbourIndex = reviewerDF[reviewerDF['ID']==NeighboursList[j]].index.values.astype(int)[0] #index of that element in the dataframe
        neighbourArray[j][5] = float(reviewerDF['AvgPrice'][neighbourIndex])  #last element is the avg price of neighbour
        if 'Grocery & Gourmet Food' in reviewerDF['AllCategories'][neighbourIndex] :
            neighbourArray[j][1]  = reviewerDF['AllCategories'][neighbourIndex].count('Grocery & Gourmet Food')   #how many items has the neighbour bought in this category
        if 'Movies & TV' in reviewerDF['AllCategories'][neighbourIndex] :
            neighbourArray[j][2]  = reviewerDF['AllCategories'][neighbourIndex].count('Movies & TV')
        if 'Toys & Games' in reviewerDF['AllCategories'][neighbourIndex] :
            neighbourArray[j][3]  = reviewerDF['AllCategories'][neighbourIndex].count('Toys & Games')    
        if 'Video Games' in reviewerDF['AllCategories'][neighbourIndex] :
            neighbourArray[j][4]  = reviewerDF['AllCategories'][neighbourIndex].count('Video Games')
            
        j+=1
        
    return neighbourArray


# In[10]:


#updating each one fron 'Person' to 'Neighbour'

def normalize_and_send_array(neighbourArray, onto, NeighboursList):
    
    import numpy as np
    from sklearn.preprocessing import normalize
    
    i=0
    for row in neighbourArray:
        row_id  = row.pop(0)
        row = np.asarray(row)
        row = row/np.linalg.norm(row)   
        row = row.tolist()
        row  = ' '.join([str(elem) for elem in row]) 
        neighbour = NeighboursList[i]
        onto.Person(neighbour, NeighbourFeatures = [row])
        i = i+1
    
    onto.save()
    
    return onto


# In[11]:


#Neighbour 

def create_neighbour_feature_dataframe(onto, active_user):
    
    import pandas as pd

    neighbourList = set(onto.Person(active_user).hasNeighbours)  #these values will be printed as 'IntegratedOntology.ABCBDBJDUEBIE'
    
    #before modifying neighbourList, make required changes and save it in newnNeighbourList for future use
    newAmazonOntology = onto
    newNeighbourList = []
    for neighbour in neighbourList:
        neighbour = str(neighbour)
        neighbour = neighbour[19:]    #want to remove the "IntegratedOntology" part of each neighbour 
        newNeighbourList.append(neighbour)
        
    neighbourFeatureList = []   #create a list of features for each neighbour

    for neighbour in neighbourList :
        featureList = onto.Person(neighbour).NeighbourFeatures
        for feature in featureList:
            feature = feature.split()
            feature[0] = float(feature[0])
            feature[1] = float(feature[1])
            feature[2] = float(feature[2])
            feature[3] = float(feature[3])
            feature[4] = float(feature[4])
            neighbourFeatureList.append(feature)
    
    neighbourFeatureDF = pd.DataFrame(neighbourFeatureList)
    neighbourFeatureDF['ID'] = newNeighbourList

    return newNeighbourList, neighbourFeatureList, neighbourFeatureDF


# In[12]:


def fit_neighbour_data(neighbourFeatureList):
    
    import numpy as np
    from sklearn.neighbors import NearestNeighbors
    
    if(len(neighbourFeatureList)>=15):
        X = np.array(neighbourFeatureList)
        knn = NearestNeighbors(n_neighbors=15)  
        knn.fit(X)
    else:
        X = np.array(neighbourFeatureList)
        knn = NearestNeighbors(n_neighbors=len(neighbourFeatureList))
        knn.fit(X)
         
    return knn


# In[13]:


#features for the active user

def create_user_features(reviewerDF, activeIndex):
    
    import numpy as np
     
    userFeature = []  #will have the number of items bought from each category

    
    #if 'Grocery & Gourmet Food' in reviewerDF['AllCategories'][activeIndex] :
    userFeature.append(reviewerDF['AllCategories'][activeIndex].count('Grocery & Gourmet Food'))
    #if 'Amazon Home' in reviewerDF['AllCategories'][activeIndex] :
    userFeature.append(reviewerDF['AllCategories'][activeIndex].count('Movies & TV'))
    #if 'Movies & TV' in reviewerDF['AllCategories'][activeIndex] :
    userFeature.append(reviewerDF['AllCategories'][activeIndex].count('Toys & Games'))
    #if 'Movies & TV' in reviewerDF['AllCategories'][activeIndex] :
    userFeature.append(reviewerDF['AllCategories'][activeIndex].count('Video Games'))


    userFeature.append(float(reviewerDF['AvgPrice'][activeIndex]))
    userFeature = np.array(userFeature)
    userFeature= userFeature / np.linalg.norm(userFeature) #need to normalise this array also

    return userFeature


# In[14]:


def run_knn_neighbour(knn, userFeature, neighbourFeatureDF):
    
    nearestNeighbourIndexArray = knn.kneighbors(userFeature.reshape(1,-1), return_distance=False)
    nearestNeighbourIndexArray = nearestNeighbourIndexArray.flatten()
    
    orderedNeighbours = []
    for ind in nearestNeighbourIndexArray:
        orderedNeighbours.append(neighbourFeatureDF['ID'][ind])
    
    return orderedNeighbours


# In[15]:


#get all the products by all the neighbours

def all_products(orderedNeighbours, onto, active_product, productDF):
    

    productList = []
    
    for neighbour in orderedNeighbours:
        products = onto.Person(neighbour).hasBoughtProducts
        for product in products:
            product = str(product) 
            product = product[19:]   #as with neighbours, had to remove the first couple letters pertaining to ontology name
            productList.append(product)            
    
    productList = list(dict.fromkeys(productList))
    
    active_product_index = productDF[productDF['ID']==active_product].index.values.astype(int)[0]
    activeProductPrice = productDF['Price'][active_product_index]
    minAllowablePrice=0.5*activeProductPrice

    
    newProductList = []
    for element in productList: 
        elementIndex = productDF[productDF['ID']==element].index.values.astype(int)[0]
        elementPrice = productDF['Price'][elementIndex]
        if(float(elementPrice) >= float(activeProductPrice)):
            newProductList.append(element)
        elif(float(elementPrice)< float(minAllowablePrice)):
            newProductList.append(element)
        elif(productDF['Category'][elementIndex]!= productDF['Category'][active_product_index]):
            newProductList.append(element)
        elif(elementIndex==active_product_index):
            newProductList.append(element)

    finalProductList = list(set(productList) - set(newProductList))
    
    if not finalProductList:
        return 0
    
    print("this is the final list")
    print(finalProductList)
               
    return finalProductList


# In[16]:


def create_product_feature_dataframe(productList, onto):

    import pandas as pd
    
    productFeatureList = []

    for product in productList :
        featureList = onto.Product(product).ProductFeatures
        for feature in featureList:
            feature = feature.split(',')
            feature[0] = float(feature[0])
            feature[1] = float(feature[1])
            productFeatureList.append(feature)
    
    for index in range(len(productFeatureList)):
        element = productFeatureList[index]
        element = [float(i)/sum(element) for i in element]
        productFeatureList[index] = element
   
    productFeatureDF = pd.DataFrame(productFeatureList)  # is created for later
    productFeatureDF['ID'] = productList
    
    return productFeatureList, productFeatureDF


# In[17]:


def fit_feature_data(productFeatureList):
    
    import numpy as np
    from sklearn.neighbors import NearestNeighbors

    X = np.array(productFeatureList)
    productKNN = NearestNeighbors(n_neighbors=3)
    try:
        productKNN.fit(X)
        return productKNN
    except ValueError:
        return -1


# In[18]:


def create_user_product_features(productDF, active_product):
    
    activeIndex = productDF[productDF['ID']==active_product].index.values.astype(int)[0]
    
    activeUserProductFeatures = productDF['Features'][activeIndex]
    activeUserProductFeatures = list(activeUserProductFeatures.split(','))
    activeUserProductFeatures[0] = float(activeUserProductFeatures[0])
    activeUserProductFeatures[1] = float(activeUserProductFeatures[1])
    activeUserProductFeatures = [float(i)/sum(activeUserProductFeatures) for i in activeUserProductFeatures]
   
    return activeIndex, activeUserProductFeatures


# In[19]:


def run_product_knn(activeUserProductFeatures, activeIndex, productKNN, productFeatureDF, productDF):
    
    import pandas as pd
    import numpy as np
    
    active_product = productDF['ID'][activeIndex]
    
    activeUserProductFeatures = np.array(activeUserProductFeatures)
    try:
        nearestProductIndexArray = productKNN.kneighbors(activeUserProductFeatures.reshape(1,-1), return_distance=False)
        nearestProductIndexArray = nearestProductIndexArray.flatten()
    
        orderedProducts = []
        for ind in nearestProductIndexArray:
               orderedProducts.append(productFeatureDF['ID'][ind])
    
    #KNN also returns the product itself
        if (active_product in orderedProducts):
               orderedProducts.remove(active_product)    
    
        return orderedProducts
    except ValueError:
        return -1
   


# In[20]:


def fetch_product_name(orderedProducts,onto):
    
    allProductInfoList = []
    for item in orderedProducts:
        productInfoList = []
        productInfoList.append(onto.Product(item).ProductID)
        productInfoList.append(onto.Product(item).ProductName)
        productInfoList.append(onto.Product(item).ProductPrice)
        allProductInfoList.append(productInfoList)

    return allProductInfoList


# In[21]:


def resetting_user(onto, active_user, reviewerDF):
    
    import owlready2 as ol
    
    #destroying the current active user from onto to recommend products for other users
    ol.destroy_entity(onto.Person(active_user))
    onto.save()
        
    #adding the previous active_user asyet another reviewer 
    personID = str(onto.Person(active_user))[19:]
    reviewerDF['AvgPrice'] = reviewerDF['AvgPrice'].astype(str)
    
    for ind in reviewerDF.index: 
        if(reviewerDF['ID'][ind]== personID):
            testPerson = onto.Person(reviewerDF['ID'][ind], 
                             ReviewerName = [reviewerDF['Name'][ind]],
                             AveragePrice = [reviewerDF['AvgPrice'][ind]],
                             ReviewerID = [reviewerDF['ID'][ind]])
    
            for product in reviewerDF['ProductIDs'][ind]:
                ontoProd = onto.Product(product)
                testPerson.hasBoughtProducts.append(ontoProd)
    
   
    onto.save()
    
    return 0


# In[22]:


#start the timer for measuring query latency
timestamp = start_timer()


# In[23]:


def run_recommendation_for_input(personOfInterest, productOfInterest):
    
    #all function calls to do with loading the precreated datsets and ontology
    productDF, reviewerDF = load_datasets()
    print("The product and reviewer dataframes have been loaded")
     
    onto = load_ontology()
    print("The ontology has been loaded")
    
    ##all function calls related to setting test data info
    active_user, active_product, activeIndex = user_info(reviewerDF, personOfInterest, productOfInterest)
    if(activeIndex):
        print("User information has been loaded")
    else:
        print("reco returns 1")
        return 1
    
    #all function calls to do with updating 'Person' to 'User'
    NeighboursList = create_neighbour_list(reviewerDF, activeIndex, active_product)
    if(NeighboursList==0):
        print("reco returns 2")
        return 2
    else:
        print('The list of all neighbours has been created')
        # print(NeighboursList)

    minPriceActiveUser, maxPriceActiveUser = pick_min_max_price(reviewerDF, activeIndex)
    print('The minimum and max purchase price of the user has been picked up')
    # print(minPriceActiveUser, maxPriceActiveUser)

    onto = update_in_ontology(onto, active_user, minPriceActiveUser, maxPriceActiveUser, NeighboursList)
    print("The active viewer has been updated to 'User from 'Person'")
    
    
    #all function calls to do with Neighbour KNN
    neighbourArray = neighbour_feature_creation(NeighboursList, reviewerDF)
    print('An array to do with each neighbour has been created')
    # for row in neighbourArray: 
    #       print(row) 
        
    onto = normalize_and_send_array(neighbourArray, onto, NeighboursList)
    print('The individual neighbour profiles have been updated with the respective features')

    newNeighbourList, neighbourFeatureList, neighbourFeatureDF = create_neighbour_feature_dataframe(onto, active_user)
    if(newNeighbourList==0):
        resetting_user(onto, active_user, reviewerDF)
        print("reco returns 2")
        return 2
    else:
        print('Have combined all the neighburs features into a dataframe')
        # print(newNeighbourList)
        # print(neighbourFeatureList)
        # print(neighbourFeatureDF)

    knn = fit_neighbour_data(neighbourFeatureList)
    print("A KNN model for the neighbour data has been created")
    # print(knn)

    userFeature = create_user_features(reviewerDF, activeIndex)
    print("The active user's feature has been created")
    # print(userFeature)

    orderedNeighbours = run_knn_neighbour(knn, userFeature, neighbourFeatureDF)
    print("The top neighbours for the given user have been created")
    # print(orderedNeighbours)
    
   
    # all function calls pertaining to product knn
    productList = all_products(orderedNeighbours, onto, active_product, productDF)
    if(productList==0):
        if(resetting_user(onto, active_user, reviewerDF)==0):
            print("Data has been reset")
        print("reco returns 2")
        return 2
    else:
        print("The list of all products purchased by all neighbours has been created ")
        # print(productList)

    if(len(productList) >=4):
        productFeatureList, productFeatureDF = create_product_feature_dataframe(productList, onto)
        print("The features for each product have been created")
        # print(productFeatureList)
        productKNN = fit_feature_data(productFeatureList)
        print("The KNN model for products has been created")
        # print(productKNN)
        activeIndex, activeUserProductFeatures = create_user_product_features(productDF, active_product)
        print("The features of the interested product has been created")
        # print(activeUserProductFeatures)
        orderedProducts = run_product_knn(activeUserProductFeatures, activeIndex, productKNN, productFeatureDF, productDF)
        # print(orderedProducts)
        allProductsInfoList = fetch_product_name(orderedProducts,onto)
        
    else:
        # print(productList)
        orderedProducts = productList
        allProductsInfoList = fetch_product_name(orderedProducts,onto)
        
    
    if(resetting_user(onto, active_user, reviewerDF)==0):
        print("Data has been reset")

    print("product list below")
    print(allProductsInfoList)    
    return allProductsInfoList    
    #run this line for analysis puurposes   
    #return allProductsInfoList, orderedProducts, onto, activeUserProductFeatures, orderedNeighbours, neighbourFeatureDF, userFeature
    


# In[24]:


# allProductsInfoList = run_recommendation_for_input('A1055QXUA6BOEL', 'B00000I1C0')
# print(allProductsInfoList)
    


# In[ ]:





# #QueryLatency
# QueryLatency = stop_timer(timestamp)
# print("Query Latency:")
# print(QueryLatency,"ms")
# 

# #Cosine Similarity (item-based)
# from scipy import spatial
# import statistics
#        
# orderedProductFeatureList, ordererdProductFeatureDF = create_product_feature_dataframe(orderedProducts, onto)
#        
# cos_sim_item = []
# for productFeatures in orderedProductFeatureList:
#                cos_sim_item.append(1 - spatial.distance.cosine(productFeatures, activeUserProductFeatures))
#        # Note: spatial.distance.cosine computes the distance, and not the similarity.
#        # So, you must subtract the value from 1 to get the similarity.  
# print("Item-based cosine similarity:")
# print(statistics.mean(cos_sim_item))

# Item-based cosine similarity:
# 0.999514477724626

# #Cosine Similarity (user-based)
# from scipy import spatial
# import statistics
# 
# cos_sim_user = []
# for neighbour in orderedNeighbours:
#     for ind in neighbourFeatureDF.index:
#         if(neighbourFeatureDF['ID'][ind]  == neighbour):
#             neighbourFeatures = neighbourFeatureDF.loc[ind, :].values.tolist()[:5]
#             cos_sim_user.append(1 - spatial.distance.cosine(neighbourFeatures, userFeature))
# 
# print("User-based cosine similarity:")
# print(statistics.mean(cos_sim_user))
# 
# 

# User-based cosine similarity:
# 1.0

# #Case 1 : Normal customer
# allProductsInfoList = run_recommendation_for_input('A1055QXUA6BOEL', 'B00000I1C0')
# print(allProductsInfoList)
# 
# 
# The product and reviewer dataframes have been loaded
# The ontology has been loaded
# User information has been loaded
# The list of all neighbours has been created
# The minimum and max purchase price of the user has been picked up
# The active viewer has been updated to 'User from 'Person'
# An array to do with each neighbour has been created
# The individual neighbour profiles have been updated with the respective features
# Have combined all the neighburs features into a dataframe
# A KNN model for the neighbour data has been created
# The active user's feature has been created
# The top neighbours for the given user have been created
# The list of all products purchased by all neighbours has been created 
# The features for each product have been created
# The KNN model for products has been created
# The features of the interested product has been created
# Data has been reset
# [[['B000007NJC'], ['WWF Warzone'], ['29.97']], [['B00000DMB5'], ['Turok 2: Seeds of Evil'], ['24.99']], [['B00000IODY'], ['Syphon Filter'], ['44.25']]]

# #Case 2: New customer
# allProductsInfoList = run_recommendation_for_input('AAAAA', 'B00000I1C0')
# print(allProductsInfoList)
# 
# 
# The product and reviewer dataframes have been loaded
# The ontology has been loaded
# Hi! Looks like you're a new user! Give us some information about yourself for better recommendations!

# #Case 3 : No neighbours found ----since this does not exist in the dataset chosen, should run as usual
# allProductsInfoList = run_recommendation_for_input('A7B8TC8T4ZOCW', '0767020294')
# print(allProductsInfoList)
# 
# 
# 
# The product and reviewer dataframes have been loaded
# The ontology has been loaded
# User information has been loaded
# The list of all neighbours has been created
# The minimum and max purchase price of the user has been picked up
# The active viewer has been updated to 'User from 'Person'
# An array to do with each neighbour has been created
# The individual neighbour profiles have been updated with the respective features
# Have combined all the neighburs features into a dataframe
# A KNN model for the neighbour data has been created
# The active user's feature has been created
# The top neighbours for the given user have been created
# The list of all products purchased by all neighbours has been created 
# The features for each product have been created
# The KNN model for products has been created
# The features of the interested product has been created
# Data has been reset
# [[['0764004581'], ['Hell in the Pacific VHS'], ['5.97']], [['0767803434'], ['Air Force One'], ['6.88']], [['0767819462'], ['Stepmom VHS'], ['0.88']]]
# 

# #Case 4 : Neighbour doesn't have anything in this category ---- should be filler line
# allProductsInfoList = run_recommendation_for_input('A27FYPCKNT58N','0005419263')  
# print(allProductsInfoList)
# 
# 
# 
# The product and reviewer dataframes have been loaded
# The ontology has been loaded
# User information has been loaded
# The list of all neighbours has been created
# The minimum and max purchase price of the user has been picked up
# The active viewer has been updated to 'User from 'Person'
# An array to do with each neighbour has been created
# The individual neighbour profiles have been updated with the respective features
# Have combined all the neighburs features into a dataframe
# A KNN model for the neighbour data has been created
# The active user's feature has been created
# The top neighbours for the given user have been created
# Data has been reset
# Hi! Looks like you've found a great product!

# #Case 5 : No cheaper product ----- should give a filler line
# allProductsInfoList = run_recommendation_for_input('AIYTOK86QCYSV', '0486448789')
# print(allProductsInfoList)
# 
# 
# 
# The product and reviewer dataframes have been loaded
# The ontology has been loaded
# User information has been loaded
# The list of all neighbours has been created
# The minimum and max purchase price of the user has been picked up
# The active viewer has been updated to 'User from 'Person'
# An array to do with each neighbour has been created
# The individual neighbour profiles have been updated with the respective features
# Have combined all the neighburs features into a dataframe
# A KNN model for the neighbour data has been created
# The active user's feature has been created
# The top neighbours for the given user have been created
# Data has been reset
# Hi! Looks like you've found a great product!

# #Case 6 : Less than 4 products
# allProductsInfoList = run_recommendation_for_input('A14G4SXIZLW63W', '0735351228')
# print(allProductsInfoList)
# 
# 
# 
# The product and reviewer dataframes have been loaded
# The ontology has been loaded
# User information has been loaded
# The list of all neighbours has been created
# The minimum and max purchase price of the user has been picked up
# The active viewer has been updated to 'User from 'Person'
# An array to do with each neighbour has been created
# The individual neighbour profiles have been updated with the respective features
# Have combined all the neighburs features into a dataframe
# A KNN model for the neighbour data has been created
# The active user's feature has been created
# The top neighbours for the given user have been created
# The list of all products purchased by all neighbours has been created 
# Data has been reset
# [[['0963469150'], ['Quiddler Card Game'], ['11.57']]]

# In[ ]:




