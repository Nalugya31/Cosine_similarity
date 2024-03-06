#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


# In[3]:


import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk.tokenize import word_tokenize 

plt.rcParams['figure.figsize'] = (13, 7)


# In[4]:


#take a look on the dataset
df = pd.read_csv("random_data.csv")
df.head(5)



# In[5]:


#showing info for each featrue
df.info()



# In[6]:


df.head(1)



# In[7]:


#now checking the NaN values
df.isnull().sum()



# In[12]:


df['Skill'] = df['Skill'].fillna("Unknown")
df['Education'] = df['Education'].fillna("Unknown")
df['Experience'] = df['Experience'].fillna("Unknown")
df = df.reset_index( drop=True)


# In[13]:


df.head(5)



# In[14]:


overall_infos = []
for i in range(0, df.shape[0]):
    overall_infos.append(df['Name'][i]+' '+df['Skill'][i]+' '+df['Education'][i]+' '+df['Experience'][i])
df['overall_infos'] = overall_infos


# In[15]:


df_new = df[['overall_infos']]
df_new.head(2)


# In[16]:


#Stopwords help us to get rid of unwanted words like: a, an, are, is, ...
from nltk.corpus import stopwords
stop = stopwords.words('english')


# In[17]:


def text_preprocessing(column):
    #make all words with lower letters
    column = column.str.lower()
    #getting rid of any punctution
    column = column.str.replace('http\S+|www.\S+|@|%|:|,|', '', case=False)
    #spliting each sentence to words to apply previous funtions on them 
    word_tokens = column.str.split()
    keywords = word_tokens.apply(lambda x: [item for item in x if item not in stop])
    #assemble words of each sentence again and assign them in new column
    for i in range(len(keywords)):
        keywords[i] = " ".join(keywords[i])
        column = keywords

    return column


# In[18]:


df_new.loc[:, 'cleaned_infos'] = text_preprocessing(df_new['overall_infos'])




# In[19]:


df['overall_infos']



# In[20]:


df_new['cleaned_infos']


# In[21]:


from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

CV = CountVectorizer()
converted_metrix = CV.fit_transform(df_new['cleaned_infos'])


# In[22]:


cosine_similarity = cosine_similarity(converted_metrix)



# In[23]:


cosine_similarity


# In[24]:


#finding the correct name of a movie
df[df['Skill'].str.contains('Finance')]


# In[25]:


get_ipython().system('pip install torch')

import torch
from datasets import load_dataset
from transformers import DistilBertTokenizer, DistilBertModel
from sklearn.metrics.pairwise import cosine_similarity


# In[35]:


import re

def clean_text(text):
    # Remove special characters and non-alphanumeric characters
    cleaned_text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    # Convert to lowercase
    cleaned_text = cleaned_text.lower()
    return cleaned_text

# Example job descriptions
job_descriptions = [
    '''
Duties and Responsibilities

Keep custody of all Board and Senior Management papers, documents and minutes.
Maintain a diary of meetings for Board, Committees and Senior Management.
Take minutes of Board meetings.
Prepare attendance registers of Board meetings and allowance requisition and reimbursement for payments to Board.
Prepare schedules of information that the Board has requested from management.
Prepare schedules and information management needs to submit to Board for consideration.
Manage all correspondence and records to and from the Executive Director.
Review documentation to Executive Director and route these to their respective department heads for action.
Prepare schedules of action points that require the attention of the Executive Director.
Read reports/proposals to Executive Director that require her input, and provide a summary of key issues that require the attention of the Executive Director.
Make travel arrangements for the Executive Director and Board members;
Maintain a diary of national and international events that require the participation of the Executive Director and Board Members.
Assist in the preparation, monitoring and managing costs related to Board budget and the Executive Director’s office.
Make travel arrangements for the strategic partners visiting the organization’s field operations through the Executive Director’s office.
Qualifications, Skills, and Experience

Must hold at least a Bachelor’s degree in any relevant discipline from a recognized university of recognized standing.
A post-graduate qualification in a management-related discipline will be an added advantage.
5 years’ work experience in a supervisory or administrative or Executive Assistant role in a private or public sector organization.
Good organizational, attention to detail, and coordination skills, ability to work calmly under the stress of conflicting deadlines and assignments.
Ability to work under tight deadlines with minimal supervision.
Effective oral & written communication and presentation skills.
Computer Skills (Excel, PowerPoint and Word Processing).
Desirable interpersonal and communication skills.
Analytical – ability to comprehend management and Board papers and other correspondences.
Organizational skills– prioritizing, time management and ability to plan each day to meet deadlines.
Strong knowledge of the Labour laws of Uganda.
High levels of integrity, professionalism and discretion.
High levels of self-initiative and problem-solving skills.
Courtesy, tact and diplomacy.





'''
]

# Apply the cleaning function to each job description
cleaned_descriptions = [clean_text(desc) for desc in job_descriptions]

# Display the cleaned job descriptions
for original, cleaned in zip(job_descriptions, cleaned_descriptions):
    print(f"Original:\n{original}\nCleaned:\n{cleaned}\n")


# In[36]:


import torch
from transformers import DistilBertTokenizer, DistilBertModel

# Assuming df_new['cleaned_infos'][4] contains a list of resumes
resumes = df_new['cleaned_infos'][4]


# Initialize the DistilBERT tokenizer and model
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
model = DistilBertModel.from_pretrained('distilbert-base-uncased')

# Tokenize and embed job descriptions
job_description_embeddings = []
for description in job_descriptions:
    tokens = tokenizer(description, padding=True, truncation=True, return_tensors='pt')
    with torch.no_grad():
        output = model(**tokens)
    embeddings = output.last_hidden_state.mean(dim=1).numpy()
    job_description_embeddings.append(embeddings[0])  # Flatten the embeddings to 1D

# Tokenize and embed resumes
resume_embeddings = []
for resume in resumes:
    tokens = tokenizer(resume, padding=True, truncation=True, return_tensors='pt')
    with torch.no_grad():
        output = model(**tokens)
    embeddings = output.last_hidden_state.mean(dim=1).numpy()
    resume_embeddings.append(embeddings[0])  # Flatten the embeddings to 1D


# In[37]:


# Calculate cosine similarity between job descriptions and resumes
similarity_scores = cosine_similarity(job_description_embeddings, resume_embeddings)
similarity_scores


# In[38]:


# Rank candidates for each job description based on similarity scores
num_top_candidates = 5
top_candidates = []

for i, job_description in enumerate(cleaned_descriptions):
    candidates_with_scores = list(enumerate(similarity_scores[i]))
    candidates_with_scores.sort(key=lambda x: x[1], reverse=True)
    top_candidates_for_job = candidates_with_scores[:num_top_candidates]
    top_candidates.append(top_candidates_for_job)

# Print the top candidates for each job description
for i, job_description in enumerate(cleaned_descriptions):
    print(f"Top candidates for JD {i+1}")
    for candidate_index, score in top_candidates[i]:
        print(f"  Candidate {candidate_index + 1} - Similarity Score: {score:.4f}")
        # Print additional information if needed
        # print(f"  Resume: {resumes[candidate_index]}")
    print()


# In[ ]:





# In[ ]:




