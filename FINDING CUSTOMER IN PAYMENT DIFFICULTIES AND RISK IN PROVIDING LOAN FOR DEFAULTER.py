#!/usr/bin/env python
# coding: utf-8

# # Credit EDA

# In[14]:


# Importing all necessary libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

get_ipython().run_line_magic('matplotlib', 'inline')
import warnings
warnings.filterwarnings('ignore')


# In[15]:


# Reading() data set

df=pd.read_csv("application_data.csv")


# In[16]:


# Determining the shape of the dataset

df.shape


# In[17]:


# Identifying the variables dtypes

df.dtypes


# # Data Cleaning And Manipulation

# In[18]:


# Checking nul values in the data frames

df.isnull().sum()


# In[23]:


# Identifying the column having more than 30% of null values

nullColumns = df.isnull().sum()
nullColumns = nullColumns[nullColumns.values > (0.3*len(nullColumns))]
len(nullColumns)


# There are 64columns in which there are more than 30% of null values. These will impact our analysis. we can drop this columns for better results.

# In[26]:


# Removing the columns having more than 30% of null values

df.drop(labels=list(nullColumns.index),axis=1,inplace=True)

df.shape


# In[27]:


# Checkig the existance of null values in the remaining dataframe

df.isnull().sum()


# "AMT ANNUITY" column has few null values, hence we try to impute them with the suitable value.

# In[28]:


df.AMT_ANNUITY.describe()


# In[29]:


# Ploting Histogram for AMT_ANNUITY column

plt.hist(df.AMT_ANNUITY)
plt.show()


# In[31]:


# Plotting Box plot to identify outliers

sns.boxplot(df.AMT_ANNUITY)
plt.show()


# Since "AMT ANNUITY" column is having an outliers which is very large, imputing missing values with mean will be inappropriate. Hence, Median comes to rescue for this and we will fill those missing values with median values.

# In[32]:


# Calculating median and replace null values with median

medianvalue = df.AMT_ANNUITY.median()
df.loc[df['AMT_ANNUITY'].isnull(),'AMT_ANNUITY'] = medianvalue


# In[33]:


# Checking the existance of null values in the remaining data frame (in percentages)

df.isnull().sum()


# In[34]:


# Reading the column names

df.columns


# In[36]:


# We will drop wanted columns from the data frame for the better analysis:

canDrop = ['FLAG_MOBIL', 'FLAG_EMP_PHONE', 'FLAG_WORK_PHONE', 'FLAG_CONT_MOBILE',
       'FLAG_PHONE', 'FLAG_EMAIL','CNT_FAM_MEMBERS', 'REGION_RATING_CLIENT',
       'REGION_RATING_CLIENT_W_CITY','DAYS_LAST_PHONE_CHANGE', 'FLAG_DOCUMENT_2', 
       'FLAG_DOCUMENT_3','FLAG_DOCUMENT_4', 'FLAG_DOCUMENT_5', 'FLAG_DOCUMENT_6',
       'FLAG_DOCUMENT_7', 'FLAG_DOCUMENT_8', 'FLAG_DOCUMENT_9','FLAG_DOCUMENT_10', 
       'FLAG_DOCUMENT_11', 'FLAG_DOCUMENT_12', 'FLAG_DOCUMENT_13', 'FLAG_DOCUMENT_14', 
       'FLAG_DOCUMENT_15','FLAG_DOCUMENT_16', 'FLAG_DOCUMENT_17', 'FLAG_DOCUMENT_18',
       'FLAG_DOCUMENT_19', 'FLAG_DOCUMENT_20', 'FLAG_DOCUMENT_21']


df.drop(labels=canDrop,axis=1,inplace =True)

df.shape


# # CHECKING THE DATATYPE OF THE COLUMN

# In[37]:


df.info(verbose= True)


# ### verifying if the object type column are correct , if these columns are incorrect we will fix them first before our analysis.

# In[40]:


# Name_ Contract_Type

df.NAME_CONTRACT_TYPE.head(10)


# In[42]:


# CODE_GENDER
df.CODE_GENDER.value_counts()


# # There are XNA values in 4 columns which means they are not available. since there are more female we can impute them with "F", this will not have any impact on our analysis.

# In[44]:


# Updating the column 'CODE_GENDER' with "F" in the dataframe

df.loc[df['CODE_GENDER']=='XNA', 'CODE_GENDER']='F'
df['CODE_GENDER'].value_counts()


# In[45]:


df.ORGANIZATION_TYPE.value_counts(normalize=True)*100


# ### 18% values in thr "ORGANIZATION_TYPE" column has XNA values, we can drop these rows from the dataframe causing no impact on analysis

# In[46]:



# Dropping XNA rows for data frame im ORGANIZATION_TYPE column

df= df[~(df.ORGANIZATION_TYPE=="XNA")]


# In[47]:


df.shape


# In[48]:


df.columns


# In[49]:


# Typecasting all the int/float variables to numeric in the dataset

toNumeric =['TARGET','CNT_CHILDREN','AMT_INCOME_TOTAL','AMT_CREDIT','AMT_ANNUITY',
                 'REGION_POPULATION_RELATIVE','DAYS_BIRTH','DAYS_EMPLOYED',
                 'DAYS_REGISTRATION','DAYS_ID_PUBLISH','HOUR_APPR_PROCESS_START',
                 'LIVE_REGION_NOT_WORK_REGION', 'REG_CITY_NOT_LIVE_CITY',
                 'REG_CITY_NOT_WORK_CITY', 'LIVE_CITY_NOT_WORK_CITY']
df[toNumeric] = df[toNumeric].apply(pd.to_numeric)


# In[50]:


df.head(10)


# #### Since we have cleaned the data set and handled the missing values, we will start our analysis 

# ### BINNING THE CATEGORICAL VALUE

# ### Lets start with categorising based on annual income

# In[52]:


# Creating bins for "AMT_INCOME_TOTAL"

bins = [0,50000,100000,150000,200000,250000,300000,350000,400000,450000,500000,10000000]
xlabels = ['0-50000','50000-100000','100000-150000','150000-200000','200000-250000',
           '250000-300000','300000-350000','350000-400000','400000-450000',
           '450000-500000','500000 and Above']

df['AMT_INCOME_RANGE'] = pd.cut(df['AMT_CREDIT'],bins=bins,labels=xlabels)


# In[54]:


# Creating bins for "AMT_CREDIT"

bins = [0,100000,200000,300000,400000,500000,600000,700000,800000,900000,1000000]
xlabels = ['0-100000','100000-200000','200000-300000','300000-400000','400000-500000','500000-600000',
           '600000-700000','700000-800000','800000-900000','900000 and Above']

df['AMT_CREDIT_RANGE'] = pd.cut(df['AMT_CREDIT'],bins=bins,labels=xlabels)


# In[56]:


# Dividing the dataset into two datasets consisting of 
# target=1 : client with payment difficulties
# Target=0 : others

df_target1 = df.loc[df["TARGET"] == 1]
df_target0 = df.loc[df["TARGET"] == 0]


# In[57]:


print("Target 1 shape : ", df_target1.shape)
print("Target 0 shape : ",df_target0.shape)


# ### There are less clients with payment difficulties(21835) compared to others (230302)

# # FINDING RHE IMBALANCE RATIO

# In[58]:


# Calculating imbalance percentage
# since the majority is target 0  and minority is target1

print("The Data Imbalance ratio is:",round(len(df_target0)/len(df_target1),2))


# # Categorical Univariate Analysis - Target 0

# In[68]:


# Common method to plot count plot

def countPlotForUnivariateAnalysis(df,col,title,hue =None):
    
    plt.rcParams["axes.labelsize"] = 10
    plt.rcParams["axes.titlesize"] = 20
    
    temCol = pd.Series(data = hue)
    fig, ax = plt.subplots()
    width = len(df[col].unique())
    fig.set_size_inches(width , 8)
    plt.xticks(rotation=90)
    plt.yscale('log') # Using log scale to capture better  analysis
    
    plt.title(title)
    ax = sns.countplot(data = df, x= col, order = df[col].value_counts().index, hue = hue)
    
    plt.show()


# In[69]:


# plotting for income range 

countPlotForUnivariateAnalysis(df_target0,col='AMT_INCOME_RANGE',title = 'Distribution of income range', hue = 'CODE_GENDER')


# ### Insights from the above graph

# 1. Income range from 1,00,000 to 1,50,000 is having more number of credits.
# 2. Credit rating for females are more than male
# 3. For 4,50,000 and above count is very less compared to others.

# In[70]:


# Plotting for Income type 

countPlotForUnivariateAnalysis(df_target0,col='NAME_INCOME_TYPE', title = 'Distribution of Income type', hue = 'CODE_GENDER')


# ### Insights from above graph

# 1. Working professionals have the highest numbers
# 2. Those awho are on Maternity leave are the least in numbers
# 3. Those who are employed in one way or the other have better results.

# In[73]:


# plotting for contract type 

countPlotForUnivariateAnalysis(df_target0,col='NAME_CONTRACT_TYPE',title='Distribution of contract type',hue='CODE_GENDER')


# ### Insights from above graph

# 1. Cash loans contracts have more credit rating than the revolving loans.
# 2. For this ,also Female is leading for applying credits.

# In[76]:


# PLOTTING FOR ORGANIZATION TYPE 
countPlotForUnivariateAnalysis(df_target0,col='ORGANIZATION_TYPE', title='Distribution of Organization type', hue= 'CODE_GENDER')


# #### Since it is difficult to interpret from the above graph we will create a graph for Organization type separately.

# In[80]:


plt.figure(figsize=(10,20))
plt.rcParams["axes.labelsize"] = 10
plt.rcParams["axes.titlesize"] = 20
plt.title("Distribution of Organization types for target :0")
plt.xticks(rotation = 90)
plt.xscale('log')
sns.countplot(data=df_target0, y='ORGANIZATION_TYPE', order= df_target0['ORGANIZATION_TYPE'].value_counts().index)


# ### Insight from above graph

# 1. 'Business entity type 3' , 'Self employed', 'Other', 'Medicine' org types have applied for more credits compared to others.
# 2. There are few clients from 'Industry type 8', 'Trade type5'

# ## Categorical Univariate Analysis - Target 1

# In[81]:


# Plotting for income range

countPlotForUnivariateAnalysis(df_target1,col="AMT_INCOME_RANGE", title= 'Distribution of income range', hue= 'CODE_GENDER')


# ###  Insights from above graph

# 1. Female counts are higher than male .
# 2. Income range from 1,00,000 to 2,00,000 is having more number of credits.
# 3. This graoh show that females are more than male in having credits for that range.

# In[83]:


# Plotting for income type

countPlotForUnivariateAnalysis(df_target1,col='NAME_INCOME_TYPE', title = 'Distribution of Income type', hue= 'CODE_GENDER')


# ### Insights from the graph

# In[87]:


countPlotForUnivariateAnalysis(df_target1,col='NAME_CONTRACT_TYPE',title='Distribution of contract type', hue = 'CODE_GENDER')


# ## Insights from graph

# 1. for contact type 'cash loans' is having a higher number of credits than 'Revolving loans' contract type.
# 2. For this reason , Women are also leading the way in applying for credits
# 3. For type 1 : there are only Female Revolving loans.

# In[93]:


plt.figure(figsize=(10,20))
plt.rcParams["axes.labelsize"] = 10
plt.rcParams["axes.titlesize"] = 20
plt.title("Distribution of Organization type for target : 1")
plt.xticks(rotation=90)
plt.xscale('log')
sns.countplot(data=df_target1,y='ORGANIZATION_TYPE',order=df_target0['ORGANIZATION_TYPE'].value_counts().index)


# ### Insights from above graph

# 1. Clients which have applied for credits are from of the organisation type 'Business entity Type 3', 'Self employed', 'Other', 'Medicine' and 'Government'.
# 2. Less cients are from Industry type 8, 6 type, 10 religion and trade type 5, type 4.
# 3. Same as type 0 in distribution of organization type.

# ### Bivariate Analysis --- Target0 ***

# In[95]:


# Box plotting for credit amount

plt.figure(figsize=(20,5))
plt.xticks(rotation=90)
sns.boxplot(data= df_target0, x='NAME_EDUCATION_TYPE', y = 'AMT_CREDIT', hue = 'NAME_FAMILY_STATUS', orient ='v')
plt.title('Credit amount vs Education Status')
plt.show()


# ### Insight from above graph

# From the above box plot we are able to conclude that family status of 'Civil marriage', 'MArriage',and 'Seprated' of academic degree education are having higher numbers of credits than other. Also , education of family status of 'marriage' , 'single' and 'civil marriage' are having more outliers. civil marriage for education degree is having most of the credtis within the third quartile.

# In[96]:


# Box plotting for income amount in lagarithmic scale 

plt.figure(figsize= (20,5))
plt.xticks(rotation = 90)
plt.yscale('log')
sns.boxplot(data = df_target0, x= 'NAME_EDUCATION_TYPE', y = 'AMT_INCOME_TOTAL', hue ='NAME_FAMILY_STATUS', orient = 'v')
plt.title('Income amount vs Education Status')
plt.show()


# ### Insights from above graph

# from above boxplot for education type ' Higher Education' on the income amount is usually equal with family status. it does contain many outliers. less outliers are having for academicsn degree but there income amount is little higher that higher education . Lower secondary of marriage family status are less income amount than others.

# ## Bivariate Analysis - Target 1***

# In[100]:


# Boxplotting for credit amount 

plt.figure(figsize=(20,5))
plt.xticks(rotation =90)
sns.boxplot(data = df_target1, x='NAME_EDUCATION_TYPE', y = 'AMT_CREDIT', hue = 'NAME_FAMILY_STATUS', orient = 'v')
plt.title('Credit amount vs Education Status')
plt.show()


# ### Insight from above graph

# From the above box plot we can say that family status of 'civil marriage' 'marriage',and 'separated' of academic degree education are having higher number of credits than others. Most of the outliers are from education type 'Higher education' and 'Secondary'. civil marriage for academics degree is having most of the credits in the third quartile.

# In[102]:


# Box plotting for income amount in logarithmic scale

plt.figure(figsize = (20, 5))
plt.xticks(rotation = 90)
plt.yscale('log')
sns.boxplot(data= df_target1, x = 'NAME_EDUCATION_TYPE', y = 'AMT_INCOME_TOTAL', hue = 'NAME_FAMILY_STATUS', orient = 'v')
plt.title('Income amount va Education Status')
plt.show()


# ### Insights from above graph

# From above box plot for education type 'Higher education' the income amount is mostly equal with family status . less outliers are having for academics degree but there income amount is little higher that Higher education . Lower secondary are have less income amount than others.
# 

# ## CORRELATION FOR THE CLIENT WITH PAYMENT DIFFICULTIES AND ALL OTHER

# In[103]:


# Find correlation between the numerical columns for target 0

df_target0_corr = df_target0.iloc[0:,2:]
target0=df_target0_corr.corr()
target0


# In[105]:


# plotting Heatmap for above correlation 
   
plt.figure(figsize=(15,10))
plt.rcParams['axes.titlesize'] = 25
   
sns.heatmap(target0, cmap = 'RdYlGn' , annot = True)
   
plt.title("Target 0")
plt.yticks(rotation = 0)
plt.show()


# ### Insights from above graph

# 1. Credit amount is inversely proportional to the date of birth , which means Credit amount is higher for low age and vice versa.
# 2. Credit amount is inversely proportional to the number of children client have, means credit amount is higher for less children count client have and vice versa
# 3. Income amount is inversely proportional to the number of children client have , means more income for less children client have and vice versa.

# In[107]:


# Find correlation between the numerical columns for target1

df_target1_corr = df_target1.iloc[0:,2:]
target1 = df_target1_corr.corr()
target1


# In[111]:


# Plotting heatmap for above correlation

plt.figure(figsize =(15, 10))
plt.rcParams['axes.titlesize'] = 25

sns.heatmap(target1,cmap="RdYlGn" ,annot =True)

plt.title(":Target 1")
plt.yticks(rotation=0)
plt.show()


# ### Insight from above graph

# 1. The client's permanent address doesnot match contact address are having less children and vice-versa.
# 2. The client's permanent address doesnot match work address are having less children and vice-versa.

# # PREVIOUS_DATA

# This data is about whether the previous application had been approved, Cancelled, Refused or Unused offer.

# ### By taking previous application into consideration for Analysis

# In[112]:


previous_df = pd.read_csv("E:\previous_application.csv")


# In[113]:


previous_df.shape


# In[114]:


# identifying and cleaning the missing values which are greater than 30%

nullColumns = previous_df.isnull().sum()
nullColumns = nullColumns[nullColumns.values > (0.3*len(nullColumns))]
len(nullColumns)


# In[115]:


# removing 15 columns

previous_df.drop(labels=list(nullColumns.index), axis=1, inplace=True)

previous_df.shape


# In[116]:


previous_df.dtypes


# In[117]:


previous_df.NAME_CASH_LOAN_PURPOSE.value_counts()


# In[119]:


# Removing the column values of 'XNA' AND 'XAP'

previous_df= previous_df[~(previous_df['NAME_CASH_LOAN_PURPOSE']=='XNA')]
previous_df= previous_df[~(previous_df['NAME_CASH_LOAN_PURPOSE']=='XAP')]

previous_df.shape


# # MERGING TWO DATAFRAMES

# In[126]:


# merging both the data frames

df = pd.merge(left = df,right = previous_df, how='inner', on = "SK_ID_CURR",suffixes='_x' )
df.shape


# In[127]:


df.columns


# In[128]:


df.dtypes


# # Univariate Analysis

# In[131]:


# Distribution of contract status

plt.figure(figsize=(11,8))
plt.rcParams["axes.labelsize"] =15
plt.rcParams['axes.titlesize'] = 20

plt.xticks(rotation = 90)
plt.xscale('log')
plt.title('Distribution of contracts with target')
ax = sns.countplot(data = df, y = 'NAME_CONTRACT_TYPEx', order = df['NAME_CONTRACT_TYPEx'].value_counts().index, hue= 'TARGET')
plt.show()


# ### Insight from above graph

# 1. Most rejection of loans came from purpose 'repairs'.
# 2. For education purposes we have equal number of approves and rejection.
# 3. Paying other loans and buying a new car is having significant higher rejection than approves.

# In[143]:


# Distribution of contracts status:
sns.set_style('whitegrid')
sns.set_context('talk')

plt.figure(figsize=(15,30))
plt.rcParams["axes.labelsize"] = 20
plt.rcParams['axes.titlesize'] = 22
plt.rcParams['axes.titlepad'] = 30
plt.xticks(rotation=90)
plt.xscale('log')
plt.title('Distribution of purposes with target ')
ax = sns.countplot(data =df, y= 'NAME_CASH_LOAN_PURPOSEx', 
                   order=df['NAME_CASH_LOAN_PURPOSEx'].value_counts().index,hue = 'TARGET')


# ### insights from above graph

# 1. Loan purpose witb 'Repairs' are facing more difficulties in paymrnt on time.
# 2. There are few places where loan payment is significant higher than facing difficulties . They aer 'Buying a garage' ,' Business development', ' Buying a new car', and 'Education'.

# ## Bivariate Analysis

# In[138]:


plt.figure(figsize=(15, 8))
plt.xticks(rotation=90)
plt.yscale('log')
sns.boxplot(data= df, x='NAME_CONTRACT_TYPEx', hue='NAME_INCOME_TYPE', y= 'AMT_CREDIT_', orient='v')
plt.title('Contract type vs amount credit')
plt.show()


# ### insights from above graph

# 1. The credit amount of loan purpose lije 'Buying a land', 'Buying a new car' and 'building a house' is higher.
# 2. Income type of state servants have a significant amount of credit applied.
# 3. Money for third person or a hobby is having less credits applied for.

# In[139]:


# Box plotting for credit amount prev vs Housing type in logarithmic sca;e

plt.figure(figsize=(16,12))
plt.xticks(rotation=90)
sns.barplot(data=df, y = 'AMT_CREDITx', hue= 'TARGET',x= 'NAME_HOUSING_TYPE')
plt.title('Prev Credit amount vs Housing type')
plt.show()


# ### Insights from above graph

# Here, for housing type , office apartment is having higher credit of target 0 and co-op apartment is having higher credit of target1. So we can conclude that the bank should avoid giving loans tot he housing type of co-op apartment as they are having difficulties in payment. Bank can focus mostly on housing type with parents or House/ apartment or municipal apartment for successful payments.
# 

# # CONCLUSION

# 1. Banks should focus more on contract type 'Student', 'Pensioner',and 'Businessman' with housing type other than 'co-op apartment' for successful payments.
# 2. Banks should focus less on income type 'working' as they are having most number of unsuccessful payments.
# 3. Also with loan purpose 'Repair' is having higher number of unsucessful payment on time.
# 4. Get as much as clients from housing type 'with parents' as they are having least number of unsucessful payments.
# 

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




