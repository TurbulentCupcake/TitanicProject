
# coding: utf-8

# In[4]:

import pandas as pd 
from pandas import Series, DataFrame


# In[5]:

titanic_df = pd.read_csv('train.csv')


# In[6]:

titanic_df.head()


# In[7]:

titanic_df.info()
#You can use the info command to check for missing info


# In[8]:

#Q1 : Who were the passengers on the titanic?


# In[9]:

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().magic(u'matplotlib inline')


# In[10]:

# We can use the factorplot to plot the count of male and females , you can just pass the column argument
sns.factorplot('Sex',data=titanic_df, kind = 'count')


# In[11]:

#seperate genders by classes
#We can use the hue parameter to seperate the data further by the subclasses
sns.factorplot('Sex', data = titanic_df,kind = 'count', hue = 'Pclass')


# In[12]:

#lets try doing the opposite
sns.factorplot('Pclass', data = titanic_df, hue = 'Sex', kind = 'count')
# here, the split is more clear, we observe there were lot more people in the 3rd class and way more males
# than females in the 3rd class


# In[20]:

def male_female_child(passenger):
    age,sex = passenger
    
    if(age < 16):
        return 'child'
    else:
        return sex
    


# In[21]:

titanic_df['person'] = titanic_df[['Age','Sex']].apply(male_female_child, axis = 1)


# In[24]:

titanic_df.head(10)


# In[25]:

#does the famous women and children policy hold during survival?


# In[26]:

sns.factorplot('Pclass', data = titanic_df, kind = 'count', hue = 'person')


# In[27]:

#lot more children in 3rd class than first 


# In[28]:

titanic_df['Age'].hist(bins = 70)


# In[29]:

titanic_df['Age'].mean()


# In[30]:

titanic_df['person'].value_counts()


# In[31]:

fig = sns.FacetGrid(titanic_df, hue = 'Sex', aspect = 4)
fig.map(sns.kdeplot,'Age',shade = True)

oldest = titanic_df['Age'].max()
fig.set(xlim = (0,oldest))
fig.add_legend()


# In[32]:

fig = sns.FacetGrid(titanic_df, hue = 'person', aspect = 4)
fig.map(sns.kdeplot,'Age',shade = True)

oldest = titanic_df['Age'].max()
fig.set(xlim = (0,oldest))
fig.add_legend()


# In[33]:

fig = sns.FacetGrid(titanic_df, hue = 'Pclass', aspect = 4)
fig.map(sns.kdeplot,'Age',shade = True)

oldest = titanic_df['Age'].max()
fig.set(xlim = (0,oldest))
fig.add_legend()


# In[34]:

#we can  use KDEplot to show how a range changes with respect to some constraint


# In[35]:


# Q2, what deck were our passengers on and how does that relate to their class?


# In[36]:

titanic_df.head()


# In[37]:

deck = titanic_df['Cabin'].dropna()


# In[38]:

deck.head()


# In[43]:

# Gets first letters in the deck
levels = []

for level in deck:
    levels.append(level[0])
    
cabin_df = DataFrame(levels)
cabin_df.columns = ['Cabin']
sns.factorplot('Cabin', data = cabin_df, palette = 'winter_d', kind = 'count')


# In[44]:

cabin_df = cabin_df[cabin_df.Cabin != 'T']


# In[49]:

sns.factorplot('Cabin', data = cabin_df, palette = 'cool', kind = 'count')


# In[50]:

titanic_df.head()


# In[51]:

#Where did the passengers come from?


# In[52]:

sns.factorplot('Embarked', data = titanic_df, hue = 'Pclass', kind = 'count', x_order = ['C','Q','S'])


# In[53]:

#From the above we can possibly raise some facts about the economics of the three towns. 


# In[54]:

#Q 4 :  Who was alone and who was with family?


# In[55]:

titanic_df.head()


# In[57]:

titanic_df['Alone'] = titanic_df.SibSp + titanic_df.Parch


# In[58]:

titanic_df['Alone']


# In[59]:

titanic_df['Alone'].loc[titanic_df['Alone'] > 0 ] = 'With Family'
titanic_df['Alone'].loc[titanic_df['Alone'] == 0 ] = 'Alone'


# In[60]:

titanic_df.head()


# In[62]:

sns.factorplot('Alone',data= titanic_df, palette = 'Blues', kind = 'count')


# In[63]:

#What factors helped someone survive the sinking of the titanic?


# In[66]:

titanic_df['Survivor'] = titanic_df.Survived.map({0:'no',1:'yes'})

sns.factorplot('Survivor', data = titanic_df, palette = ['red','blue'], kind = 'count')


# In[68]:

sns.factorplot('Pclass','Survived', hue = 'person',data=titanic_df)


# In[69]:

#Being a male decreases your chances of being a survivor, makes it worse if you were in the 3rd class


# In[70]:

sns.lmplot('Age','Survived',data = titanic_df)


# In[72]:

sns.lmplot('Age','Survived',data = titanic_df, hue = 'Pclass', palette = 'cool')


# In[73]:

generations = [10,20,40,60,80]

sns.lmplot('Age','Survived', hue = 'Pclass', data = titanic_df, palette = 'cool', x_bins = generations)


# In[74]:

sns.lmplot('Age','Survived', hue = 'Sex', data=titanic_df, x_bins = generations)


# In[75]:

# Did the deck have an effect on the passenger survival rate?
#Does your gender affect that?

#Did having a family member increase the odds of surviving a crash?


# In[ ]:



