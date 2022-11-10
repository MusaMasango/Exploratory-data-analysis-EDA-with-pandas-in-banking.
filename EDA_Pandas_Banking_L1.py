#!/usr/bin/env python
# coding: utf-8

# # **Exploratory data analysis (EDA) with Pandas in banking**
# 

# The purpose of this project is to conduct the exploratory data analysis (EDA) in banking using Pandas framework.
# 
# During this project we will do the following
# 
# 1.  Explore a banking dataset with Pandas framework.
# 2.  Build pivot tables.
# 3.  Visualize the dataset with various plot types.
# 

# The data that we are going to use for this is a subset of an open source Bank Marketing Data Set from the UCI ML repository: [https://archive.ics.uci.edu/ml/citation_policy.html](https://archive.ics.uci.edu/ml/citation_policy.html?utm_medium=Exinfluencer&utm_source=Exinfluencer&utm_content=000026UJ&utm_term=10006555&utm_id=NA-SkillsNetwork-Channel-SkillsNetworkQuickLabsEDA_Pandas_Banking_L126457256-2022-01-01).
# 
# 
# In this project, we will try to give answers to a set of questions that may be relevant when analyzing banking data:
# 
# 1.  What is the share of clients attracted in our source data?
# 2.  What are the mean values ​​of numerical features among the attracted clients?
# 3.  What is the average call duration for the attracted clients?
# 4.  What is the average age among the attracted and unmarried clients?
# 5.  What is the average age and call duration for different types of client employment?
# 
# In addition, we will make a visual analysis in order to plan marketing banking campaigns more effectively.
# 

# ## Libraries import
# 

# Import the libraries necessary for this lab.
# 

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

get_ipython().run_line_magic('matplotlib', 'inline')
plt.rcParams["figure.figsize"] = (8, 6)

import warnings
warnings.filterwarnings('ignore')


# Further specify the value of the `precision` parameter equal to 2 to display two decimal signs (instead of 6 as default).
# 

# ## Dataset exploration
# 

# In this section you will explore the sourse dataset.
# 

# Let's read the data and look at the first 5 rows using the `head` method. The number of the output rows from the dataset is determined by the `head` method parameter.
# 

# In[2]:


df = pd.read_csv('bank-additional-full.csv', sep = ';')
df.head(5)


# ### Let's look at the dataset size, feature names and their types
# 

# In[3]:


df.shape


# The dataset contains 41188 objects (rows), for each of which 21 features are set (columns), including 1 target feature (`y`).
# 

# ### Attributing information
# 

# Output the column (feature) names:
# 

# In[4]:


df.columns


# Input features (column names):
# 
# 1.  `age` - client's age in years (numeric)
# 2.  `job` - type of job (categorical: `admin.`, `blue-collar`, `entrepreneur`, `housemaid`, `management`, `retired`, `self-employed`, `services`, `student`, `technician`, `unemployed`, `unknown`)
# 3.  `marital` - marital status (categorical: `divorced`, `married`, `single`, `unknown`)
# 4.  `education` - client's education (categorical: `basic.4y`, `basic.6y`, `basic.9y`, `high.school`, `illiterate`, `professional.course`, `university.degree`, `unknown`)
# 5.  `default` - has credit in default? (categorical: `no`, `yes`, `unknown`)
# 6.  `housing` - has housing loan? (categorical: `no`, `yes`, `unknown`)
# 7.  `loan` - has personal loan? (categorical: `no`, `yes`, `unknown`)
# 8.  `contact` - contact communication type (categorical: `cellular`, `telephone`)
# 9.  `month` - last contact month of the year (categorical: `jan`, `feb`, `mar`, ..., `nov`, `dec`)
# 10. `day_of_week` - last contact day of the week (categorical: `mon`, `tue`, `wed`, `thu`, `fri`)
# 11. `duration` - last contact duration, in seconds (numeric).
# 12. `campaign` - number of contacts performed and for this client during this campaign (numeric, includes the last contact)
# 13. `pdays` - number of days that have passed after the client was last contacted from the previous campaign (numeric; 999 means the client has not been previously contacted)
# 14. `previous` - number of contacts performed for this client before this campaign (numeric)
# 15. `poutcome` - outcome of the previous marketing campaign (categorical: `failure`, `nonexistent`, `success`)
# 16. `emp.var.rate` - employment variation rate, quarterly indicator (numeric)
# 17. `cons.price.idx` - consumer price index, monthly indicator (numeric)
# 18. `cons.conf.idx` - consumer confidence index, monthly indicator (numeric)
# 19. `euribor3m` - euribor 3 month rate, daily indicator (numeric)
# 20. `nr.employed` - number of employees, quarterly indicator (numeric)
# 
# Output feature (desired target):
# 
# 21. `y` - has the client subscribed a term deposit? (binary: `yes`,`no`)
# 

# To see the general information on all the DataFrame features (columns), we use the **`info`** method:
# 

# In[5]:


print(df.info())


# As you can see, the dataset is full, no pass (`non-null`), so there is no need to fill the gaps. The dataset contains 5 integer (`int64`), 5 real (`float64`) and 11 categorical and binary (`object`) features.
# 

# Method **`describe`** shows the main statistical characteristics of the dataset for each numerical feature (`int64` and `float64` types): the existing values number, mean, standard deviation, range, min & max, 0.25, 0.5 and 0.75 quartiles.
# 

# In[6]:


df.describe()


# The `Mean` row shows the feature average, `STD` is an RMS (Root Mean Square) deviation, `min`,`  max ` - the minimum and maximum values, `25%`, `50%`, ` 75%  `- quarters that split the dataset (or part of it) into four groups containing approximately an equal number of observations (rows). For example, the duration (`duration`) of about a quarter of calls to customers is around 100 seconds.
# 

# In general, according to the data, it is impossible to say that there are outliers in the data. However, such an inspection is not enough, it is desirable to still see the charts of the target feature dependence from each input feature. We will do it later when we visualize features and dependencies.
# 

# To see the statistics on non-numeric features, you need to explicitly specify the feature types by the `include` parameter. You can also set `include = all` to output statistics on all the existing features.
# 

# In[7]:


df.describe(include = "all")


# The result shows that the average client refers to administrative staff (`job = admin.`), is married (`marital = married`) and has a university degree (`education = university.degree`).
# 

# For categorical (type `object`) and boolean (type `bool`) features you can use the **`value_counts`** method. Let's look at the target feature (`y`) distribution:
# 

# In[10]:


df["y"].value_counts()


# 4640 clients (11.3%) of 41188 issued a term deposit, the value of the variable `y` equals `yes`.
# 
# Let's look at the client distribution by the variable `marital`.
# 

# In[11]:


df["marital"].value_counts(normalize = True)


# As we can see, 61% (0.61) of clients are married, which must be taken into account when planning marketing campaigns to manage deposit operations.
# 

# ### Sorting
# 

# A `DataFrame` can be sorted by a few feature values. In our case, for example, by `duration` (`ascending = False` for sorting in descending order):
# 

# In[12]:


df.sort_values(by = "duration", ascending = False).head()


# The sorting results show that the longest calls exceed one hour, as the value `duration` is more than 3600 seconds or 1 hour. At the same time, it usually was on Mondays and Thursdays (`day_of_week`) and, especially, in November and August (`month`).
# 

# Sort by the column group:
# 

# In[13]:


df.sort_values(by = ["age", "duration"], ascending = [True, False]).head()


# We see that the youngest customers are at the `age` of 17, and the call `duration` exceeded 3 minutes only for three clients, which indicates the ineffectiveness of long-term interaction with such clients.
# 

# ### Application of functions: `apply`, `map` etc.
# 

# **Apply the function to each column:**
# 

# In[14]:


df.apply(np.max)


# The oldest client is 98 years old (`age` = 98), and the number of contacts with one of the customers reached 56 (`campaign` = 56).
# 

# The `apply` method can also be used to apply the function to each row. To do this, you need to specify the `axis = 1`.
# 

# **Apply the function to each column cell**
# 

# The `map` can also be used for **the values ​​replacement in a column** by passing it as an argument dictionary in form of ` {old_value: new_value}  `.
# 
# 

# In[15]:


d = {"no": 0, "yes": 1}
df["y"] = df["y"].map(d)
df.head()


# In[16]:


df.describe()


# ### Indexing and extracting data
# 

# A `DataFrame` can be indexed in many ways. In this regard, consider various ways of indexing and extracting data from the DataFrame with simple question examples.
# 
# You can use the code `dataframe ['name']` to extract a separate column. We use this to answer the question: **What is the share of clients attracted in our DataFrame?**
# 

# In[17]:


print("Share of attracted clients =", '{:.1%}'.format(df["y"].mean()))


# 11,3% is a rather bad indicator for a bank, with such a percentage of attracted customers a business can collapse.
# 

# Logical indexation by one column of a `DataFrame` is very convenient. It looks like this: `df [p(df['Name']]`, where`  p ` is a certain logical condition that is checked for each element of the `Name` column. The result of such an indexation is a `DataFrame` consisting only of the rows satisfying the condition `p` by the `Name` column.
# 
# We use this to answer the question: **What are the mean values ​​of numerical features among the attracted clients?**

# In[18]:


df[df["y"] == 1].mean()


# Thus, the average age of the attracted clients is about 40 (`age` = 40.91), and 2 calls were required to attract them (`campaign` = 2.05).
# 

# Combining two previous types of indexation, we will answer the question: **What is the average call duration for the attracted clients**?
# 

# In[19]:


acd = round(df[df["y"] == 1]["duration"].mean(), 2)
acd_in_min = acd // 60
print("Average call duration for attracted clients =", acd_in_min, "min", int(acd) % 60, "sec")


# So, the average duration of a successful call is almost 553 seconds, that is, nearly 10 minutes.
# 

# **What is the average age of attracted (`y == 1`) and unmarried (`'marital' == 'single'`) clients?**
# 

# In[20]:


print("Average age of attracted clients =", int(df[(df["y"] == 1) & (df["marital"] == "single")]["age"].mean()), "years")


# The average age of unmarried attracted clients is 31, which should be considered when working with such clients.
# 

# ## Pivot tables
# 

# Suppose we want to see how observations in our sample are distributed in the context of two features - `y` and `marital`. To do this, we can build **cross tabulation** by the `crosstab` method.

# In[21]:


pd.crosstab(df["y"], df["marital"])


# The result shows that the number of attracted married clients is 2532 (`y = 1` for `married`) from the total number.
# 

# In[22]:


pd.crosstab(df["y"],
            df["marital"],
            normalize = 'index')


# We see that more than half of the clients (61%, column `married`) are married and have not issued a deposit.
# 

# In `Pandas`, **pivot tables** are implemented by the method `pivot_table` with such parameters:
# 
# *   `values` – a list of variables to calculate the necessary statistics,
# *   `index` – a list of variables to group data,
# *   `aggfunc` — values that we actually need to count by groups - the amount, average, maximum, minimum or something else.
# 
# Let's find the average age and the call duration for different types of client employment `job`:
# 

# In[23]:


df.pivot_table(values=["age", "duration"],index=["job"],aggfunc = "mean").head(10)


# The obtained results allow you to plan marketing banking campaigns more effectively.
# 

# ## Visualization in Pandas
# 

# Method **scatter_matrix** allows you to visualize the pairwise dependencies between the features (as well as the distribution of each feature on the diagonal). We will do it for numerical features.
# 

# In[24]:


pd.plotting.scatter_matrix(
    df[["age", "duration", "campaign"]],
    figsize = (13, 13),
    diagonal = "kde")
plt.show()


# A scatter matrix (pairs plot) compactly plots all the numeric variables we have in a dataset against each other.
# The plots on the main diagonal allow you to visually define the type of data distribution: the distribution is similar to normal for age, and for a call duration and the number of contacts.
# 

# ## Histogram for each feature
# 

# In[25]:


df["age"].hist()
plt.title('Age distribution')
plt.xlabel('Age')


# The histogram shows that most of our clients are between the ages of 25 and 50, which corresponds to the actively working part of the population.
# 

# ## Different histograms showing the distribution of different features
# 

# In[26]:


df.hist(color = "k",
        bins = 30,
        figsize = (15, 10))
plt.show()


# [**Box Plot** ("Box and whisker plot")](https://en.wikipedia.org/wiki/box_plot?utm_medium=Exinfluencer&utm_source=Exinfluencer&utm_content=000026UJ&utm_term=10006555&utm_id=NA-SkillsNetwork-Channel-SkillsNetworkQuickLabsEDA_Pandas_Banking_L126457256-2022-01-01) is useful too. It allows you to compactly visualize the main characteristics of the feature distribution (the median, lower and upper quartile, minimal and maximum, outliers).
# 

# In[27]:


df.boxplot(column = "age",
           by = "marital")
plt.show()


# The plot shows that unmarried people are on average younger than divorced and married ones. For the last two groups, there is an outlier zone over 70 years old, and for unmarried - over 50.
# 

# **You can do this by data grouping on any other feature:**

# In[28]:


df.boxplot(column = "age",
           by = ["marital", "housing"],
           figsize = (20, 20))
plt.show()


# As you can see, age and marital status do not have any significant influence on having a housing loan.
# 

# ## Tasks
# 

# ### Question 1
# 

# List 10 clients with the largest number of contacts.
# 

# In[29]:


df.sort_values(by = "campaign", ascending = False).head(10)


# ### Question 2
# 

# Determine the median age and the number of contacts for different levels of client education.
# 

# In[30]:


df.pivot_table(values=["age", "campaign"],index=["education"],aggfunc = ["mean", "count"]).head(5)


# ### Question 3
# 

# Output box plot to analyze the client age distribution by their education level.
# 

# In[31]:


df.boxplot(column = "age",
  by = "education",
  figsize = (15, 15))
plt.show()

