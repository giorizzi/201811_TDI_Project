#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  5 13:26:27 2018

@author: giovannirizzi
"""

import pandas as pd
import numpy as np
from sqlalchemy import create_engine

engine = create_engine('postgresql://open_trial:password@localhost/open_trials_db')
query="""SELECT * from trials
Where id IN (SELECT trial_id from trials_conditions
WHERE condition_id IN (SELECT id FROM conditions
WHERE (name LIKE '%%oma %%'AND NOT name LIKE '%%hematoma%%'AND NOT name LIKE '%%glaucoma%%'AND NOT name like '%%coma%%' AND NOT name like '%%stoma%%' AND NOT name like '%%panosoma%%') 
OR   name LIKE '%%cancer%%' 
OR   name LIKE '%%tumor%%'
OR   name LIKE '%%tumour%%'));"""
             
trials = pd.read_sql_query(query, engine)
trials.drop_duplicates(subset ="id", 
                     keep = False, inplace = True)
trialsbck=trials
#%%
trials=trialsbck

#%%

query="""SELECT id,name FROM conditions
WHERE id IN (SELECT condition_id FROM trials_conditions
WHERE condition_id IN (SELECT id FROM conditions
WHERE (name LIKE '%%oma %%'AND NOT name LIKE '%%hematoma%%'AND NOT name LIKE '%%glaucoma%%'AND NOT name like '%%coma%%' AND NOT name like '%%stoma%%' AND NOT name like '%%panosoma%%') 
OR   name LIKE '%%cancer%%' 
OR   name LIKE '%%tumor%%'
OR   name LIKE '%%tumour%%'));"""

conditions = pd.read_sql_query(query, engine)

#%%
query="""SELECT trial_id, condition_id FROM trials_conditions
WHERE condition_id IN (SELECT id FROM conditions
WHERE (name LIKE '%%oma %%'AND NOT name LIKE '%%hematoma%%'AND NOT name LIKE '%%glaucoma%%'AND NOT name like '%%coma%%' AND NOT name like '%%stoma%%' AND NOT name like '%%panosoma%%') 
OR   name LIKE '%%cancer%%' 
OR   name LIKE '%%tumor%%'
OR   name LIKE '%%tumour%%');"""

trial_condition = pd.read_sql_query(query, engine)

##%% 
#condition_name=pd.Series()
#trial_condition['condition_name']=""
#a=0
#for condition_id in conditions['id']:
#    a=a+1
#    print(a)
#    location=(trial_condition['condition_id'] ==condition_id)
#    one_name=conditions.loc[conditions['id']==condition_id,'name'].item()
#    #print(one_name)
#    trial_condition.loc[location,['condition_name']]=one_name
#    #test=conditions.loc[(conditions['id'] ==conditions['id'][3])]
#%%

trial_condition=trial_condition.merge(conditions, left_on='condition_id', right_on='id', how='left',left_index=True)
trial_condition=trial_condition.drop('id',axis=1)
#%%
trials=trials.merge(trial_condition,left_on='id', right_on='trial_id', how='outer')

trials=trials.drop('trial_id',axis=1)

trials.drop_duplicates(subset ="id", keep = False, inplace = True)

#%%
years=[]
for date in trials['registration_date']:
    years.append(date.year)
trials['year']=years

#%% Let's bother later about nation, and let's go into title analysis
import matplotlib
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction import text
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cluster import KMeans
from nltk.tokenize import RegexpTokenizer
from nltk.stem.snowball import SnowballStemmer
from numpy.linalg import norm
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import random
#%%

#study_column='name'
study_column='brief_summary'
trials_NLP=trials
#trials_NLP=trials_NLP.drop_duplicates(study_column)
trials_NLP=trials_NLP.dropna(subset=[study_column])
desc = trials_NLP[study_column].values

#%%
punc = ['non','small','cell','clinical','trial','using','versus','study','rationale','goal','objective','patient','patients','therapy','treatment','treating','cancer','tumor','tumour','.', ',', '"', "'", '?', '!', ':', ';', '(', ')', '[', ']', '{', '}',"%"]
stop_words = text.ENGLISH_STOP_WORDS.union(punc)

vectorizer = TfidfVectorizer(stop_words = stop_words)

#%%
stemmer = SnowballStemmer('english')
tokenizer = RegexpTokenizer(r'[a-zA-Z\']+')

def tokenize(text):
    return [stemmer.stem(word) for word in tokenizer.tokenize(text.lower())]

#%%
vectorizer = TfidfVectorizer(stop_words = stop_words, tokenizer = tokenize, max_features = 1000)
word_matrix= vectorizer.fit_transform(desc)
words = vectorizer.get_feature_names()

#%%
kmeans = KMeans(n_clusters = 15, n_init = 5, n_jobs = -1,verbose=1,max_iter=100)
kmeans.fit(word_matrix)
trials_NLP['cluster'] = kmeans.labels_
#%%
common_words = kmeans.cluster_centers_.argsort()[:,-1:-11:-1]
for num, centroid in enumerate(common_words):
    print(str(num) + ' : ' + ', '.join(words[word] for word in centroid))


#%%
n_to_plot=5;
clusters = trials_NLP.groupby(['cluster', 'year']).size()
clusters=clusters.unstack(level = 'cluster')
clusters=clusters.fillna(0)
clusters=clusters.T/norm(clusters,axis=1,ord=1)
clusters=clusters.T
clusters.to_csv(study_column+"_clusters.csv")
common_clusters=clusters.loc[2000:2017,:].mean().sort_values(ascending=False).iloc[:n_to_plot]


#%%
plt.figure()
curves=plt.plot(clusters.loc[2000:2017,common_clusters.index],linewidth=3.0)
plt.legend(curves,np.arange(1,n_to_plot+1,1))
plt.xlabel('Year', fontsize=12, color='black')
plt.ylabel('Fraction of total trials', fontsize=12, color='black')
plt.axis([1999, 2017, 0, 0.6])
plt.savefig('Cluster_vs_time.pdf')

#%%
j=1
for i in common_clusters.index:
    print(j)
    color=curves[j-1].get_color();
    
    def grey_color_func(word, font_size, position, orientation, random_state=None,
                    **kwargs):
        h = color.lstrip('#')
        RGB=np.array([int(h[i:i+2], 16) for i in (0, 2 ,4)])
        RGB=RGB+random.randint(-100, 100)
        RGB=RGB.clip(0,255)
        return "RGB("+ str(RGB[0])+", "+ str(RGB[1])+", "+ str(RGB[2])+")"
    
    text = " ".join(review for review in trials_NLP.loc[trials_NLP['cluster']==i,study_column])
    print ("There are {} words in the combination of all review.".format(len(text)))
    
    # Create and generate a word cloud image:
    wordcloud = WordCloud(max_font_size=100, max_words=100, background_color="white", stopwords=stop_words).generate(text)
    
    # Display the generated image:
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.imshow(wordcloud.recolor(color_func=grey_color_func, random_state=3),
           interpolation="bilinear")
    plt.axis("off")
    plt.show()
    wordcloud.to_file("cluster"+str(j)+".png")
    j=j+1


#%%

study_column='public_title'
#study_column='brief_summary'
trials_NLP=trials
#trials_NLP=trials_NLP.drop_duplicates(study_column)
trials_NLP=trials_NLP.dropna(subset=[study_column])
trials_NLP = trials_NLP.reset_index(drop=True)
desc = trials_NLP[study_column].values

drug_list = pd.read_csv('drug_list.csv')
drug_vocabulary=drug_list["Name"].str.lower();
#drug_vectorizer=TfidfVectorizer(stop_words = stop_words, tokenizer = tokenize, vocabulary=drug_vocabulary)
drug_vectorizer = CountVectorizer(vocabulary=drug_vocabulary)
drug_matrix= drug_vectorizer.fit_transform(desc)
drugs = drug_vectorizer.get_feature_names()
drug_matrix=drug_matrix>0

#%%
drugs_df=pd.DataFrame(drug_matrix.todense())
drugs_df.columns=drugs
common_drugs=drugs_df.sum().sort_values(ascending=False)
common_drugs=common_drugs.reset_index()
common_drugs.columns=["name","count"]

#%%
drugs_df=pd.concat([trials_NLP[["id","year","name",study_column]],drugs_df],axis=1)

#%%
n_to_plot=5
trials_per_year=drugs_df.groupby(['year']).size()
drugs_vs_year = drugs_df.groupby(['year']).sum()
drugs_sum=drugs_vs_year.sum(axis=1)
drugs_vs_year = (drugs_vs_year.T/drugs_sum).T
plt.figure()
curves=plt.plot(drugs_vs_year.loc[2000:2017,common_drugs.loc[:n_to_plot-1,"name"]],linewidth=3.0)
plt.legend(curves,common_drugs.loc[0:n_to_plot,"name"])
plt.xlabel('Year', fontsize=12, color='black')
plt.ylabel('Relative Occurence', fontsize=12, color='black')
plt.axis([1999, 2017, 0, 0.15])
plt.savefig('drugs_vs_time.pdf')
#%%
j=1
for i in common_drugs.loc[:n_to_plot-1,"name"]:
    color=curves[j-1].get_color();
    def grey_color_func(word, font_size, position, orientation, random_state=None,
                    **kwargs):
        h = color.lstrip('#')
        RGB=np.array([int(h[i:i+2], 16) for i in (0, 2 ,4)])
        RGB=RGB+random.randint(-100, 100)
        RGB=RGB.clip(0,255)
        return "RGB("+ str(RGB[0])+", "+ str(RGB[1])+", "+ str(RGB[2])+")"
    print(str(j)+i)
    text = " ".join(review for review in drugs_df.loc[drugs_df[i]==True,"name"])
    print ("There are {} words in the combination of all review.".format(len(text)))
    
    # Create and generate a word cloud image:
    wordcloud = WordCloud(max_font_size=100, max_words=50, background_color="white", stopwords=stop_words).generate(text)
    
    # Display the generated image:
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.imshow(wordcloud.recolor(color_func=grey_color_func, random_state=3),
           interpolation="bilinear")
    font = {'family': 'sans',
        'color':  color,
        'weight': 'normal',
        'size': 25,
        }
    plt.title(i.capitalize(),font)
    plt.show()
    #plt.savefig(str(j)+i+"XX.png")
    wordcloud.to_file(str(j)+i+".png")
    j=j+1