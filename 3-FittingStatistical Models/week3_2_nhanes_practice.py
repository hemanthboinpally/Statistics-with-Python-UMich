
# coding: utf-8

# # Practice notebook for regression analysis with dependent data in NHANES
# 
# This notebook will give you the opportunity to perform some analyses
# using the regression methods for dependent data that we are focusing
# on in this week of the course.
# 
# Enter the code in response to each question in the boxes labeled "enter your code here".
# Then enter your responses to the questions in the boxes labeled "Type
# Markdown and Latex".
# 
# This notebook is based on the NHANES case study notebook for
# regression with dependent data.  Most of the code that you will need
# to write below is very similar to code that appears in the case study
# notebook.  You will need to edit code from that notebook in small ways
# to adapt it to the prompts below.
# 
# To get started, we will use the same module imports and read the data
# in the same way as we did in the case study:

# In[3]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import statsmodels.api as sm
import numpy as np

url = "nhanes_2015_2016.csv"
da = pd.read_csv(url)


# In[4]:


da.columns


# In[5]:


# Drop unused columns, drop rows with any missing values.
vars = ["BPXSY1", "RIDAGEYR", "RIAGENDR", "RIDRETH1", "DMDEDUC2", "BMXBMI",
        "SMQ020", "SDMVSTRA", "SDMVPSU",'BPXDI1']
da = da[vars].dropna()

# This is the grouping variable
da["group"] = 10*da.SDMVSTRA + da.SDMVPSU


# ## Question 1: 
# 
# Build a marginal linear model using GEE for the first measurement of diastolic blood pressure (`BPXDI1`), accounting for the grouping variable defined above.  This initial model should have no covariates, and will be used to assess the ICC of this blood pressure measure.

# In[6]:


# enter your code here
model = sm.GEE.from_formula('BPXDI1 ~ 1', groups="group",data=da)


# In[8]:


result = model.fit()
print(result.cov_struct.summary())


# In[11]:


model = sm.GEE.from_formula('BPXDI1 ~ 1', groups="group", 
                            cov_struct=sm.cov_struct.Exchangeable(),
                            data=da)
result = model.fit()
print(result.cov_struct.summary())


# __Q1a.__ What is the ICC for diastolic blood pressure?  What can you
#   conclude by comparing it to the ICC for systolic blood pressure?

# ICC of 0.03 similar to the systolic blood pressure.

# ## Question 2: 
# 
# Take your model from question 1, and add gender, age, and BMI to it as covariates.

# In[14]:


da.BMXBMI.value_counts()


# In[15]:


# enter your code here
da["RIAGENDRx"] = da.RIAGENDR.replace({1: "Male", 2: "Female"})

model = sm.GEE.from_formula('BPXDI1 ~ RIDAGEYR + RIAGENDRx + BMXBMI', groups="group", 
                            cov_struct=sm.cov_struct.Exchangeable(),
                            data=da)
result = model.fit()
print(result.cov_struct.summary())


# __Q2a.__ What is the ICC for this model?  What can you conclude by comparing it to the ICC for the model that you fit in question 1?

# ICC is 0.030 - It is almost the same as the first question. It didn't explain the intercluster variance.

# ## Question 3: 
# 
# Split the data into separate datasets for females and for males and fit two separate marginal linear models for diastolic blood pressure, one only for females, and one only for males.

# In[17]:


# enter your code here
da_males = da[da.RIAGENDRx == "Male"]


# In[18]:


da_females = da[da.RIAGENDRx == "Female"]


# In[19]:


model = sm.GEE.from_formula('BPXDI1 ~ 1', groups="group", 
                            cov_struct=sm.cov_struct.Exchangeable(),
                            data=da_males)
result = model.fit()
print(result.cov_struct.summary())


# In[20]:


model = sm.GEE.from_formula('BPXDI1 ~ 1', groups="group", 
                            cov_struct=sm.cov_struct.Exchangeable(),
                            data=da_females)
result = model.fit()
print(result.cov_struct.summary())


# __Q3a.__ What do you learn by comparing these two fitted models?

# ICC is higher for males than females

# ## Question 4: 
# 
# Using the entire data set, fit a multilevel model for diastolic blood pressure, predicted by age, gender, BMI, and educational attainment.  Include a random intercept for groups.

# In[21]:


da.DMDEDUC2.value_counts()


# In[27]:


# enter your code here
da["age_cen"] = da.groupby("group").RIDAGEYR.transform(lambda x: x - x.mean())
model = sm.MixedLM.from_formula("BPXDI1 ~ age_cen + RIAGENDRx + BMXBMI + C(DMDEDUC2)",
                                groups="group",data=da)


# In[28]:


result = model.fit()


# In[29]:


result.summary()


# __Q4a.__ How would you describe the strength of the clustering in this analysis?

# __Q4b:__ Include a random intercept for age, and describe your findings.
