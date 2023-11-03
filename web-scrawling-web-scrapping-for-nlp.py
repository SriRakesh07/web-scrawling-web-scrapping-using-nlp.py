#!/usr/bin/env python
# coding: utf-8

# # Introduction
# 
# In this, we are going to do web-scrawling (collecting links for the web-pages and web-scrapping (collecting data from the web-pages) for Natural Language Processing

# In[1]:


# import libraries

import numpy as np
import pandas as pd 
import regex as re
import itertools


# # Dependencies

# In[2]:


from bs4 import BeautifulSoup
import requests
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
from nltk.stem import WordNetLemmatizer


# # Web-scrapping and Web-Scrawling functions

# **First, let's specify the pages we are going to take texts from. I've found a site with short stories and I am going to use these texts to train a model. The categories we have, the more data we need, so we'll train a model for only two categories.**

# In[3]:


tale = "https://americanliterature.com/author/beatrix-potter/short-story/the-tale-of-peter-rabbit"
little = "https://americanliterature.com/childrens-stories/little-red-riding-hood"


# In[4]:


def get_story_link_body(link):
    storylinks = []
    hdr = {'User-Agent': 'Mozilla/5.0'}
    py_page = requests.get(link,headers=hdr)
    py_soup = BeautifulSoup(py_page.content, 'lxml')
    #print(py_soup)
    jobs = py_soup.find_all("div",class_='submission')
    storylinks.extend(jobs)
    return storylinks
print(str(get_story_link_body("https://americanliterature.com/author/beatrix-potter/short-story/the-tale-of-peter-rabbit"))[0:1000])
print(type(get_story_link_body("https://americanliterature.com/childrens-stories/little-red-riding-hood")))


# **Is this what we needed? No. Let's extract the link from the html code we have got in the previous function**

# In[5]:


html_sample = get_story_link_body('https://americanliterature.com/author/beatrix-potter/short-story/the-tale-of-peter-rabbit')
def extract_links(htmls):
    urls = []
    for h in htmls:
        a = h.find("a")
        if a!=None:
            urls.append(a['href'])
        else:
            urls.append(None)
    return urls

print(extract_links(html_sample))


# **Now we can finaly extract texts themselves!**

# In[6]:


def get_text(url):
    hdr = {'User-Agent': 'your bot 0.1'}
    py_page = requests.get(url,headers=hdr)
    print(py_page)
    py_soup = BeautifulSoup(py_page.content, 'lxml') #class_='ql-align-justify
    pre_jobs = py_soup.find_all("div",class_='writing-prompts')
    pre_jobs = [el.find_all('section',class_='row-thin row-white') for el in pre_jobs]
    strings = []
    for il in pre_jobs:
        jobs = [el.find_all("p",class_=None) for el in il]
        strings.extend(jobs)
    return strings
    
sample_story = "https://americanliterature.com/childrens-stories/jack-and-the-beanstalk"
text = get_text(sample_story)
print(type(text))
print(text)


# In[7]:


def flatten_to_string(list_html):
    flat_list = [str(el) for el in list(itertools.chain(*list_html))]
    
    return " ".join(flat_list)
text = flatten_to_string(text)


# **That's a great results! Now we have just to join and clean the text!**
# 
# P.S. May be, there is a more efficient way to do this, I am not very experienced in web-scrapping

# In[8]:


def clean_text(text_html):
    #text = str(html_body)
    #text = text[1:-1]
    text = re.sub(re.compile('<p>[\t]*[\d]+[\t]*<\/p>'),'',text_html)
    #text = re.sub(re.compile("<(\/?)?p>,"),'',text)
    text = text.replace("</p>",'')
    text = text.replace('<p>','')
    text = text.replace('<em>','')
    text = text.replace('</em>','')
    text = text.replace('xa0','')
    return text
text_2 = clean_text(text)
print(len(text_2))


# # Pt.2. Applying on larger data.
# 
# **Now we can apply these functions on multiple texts! You may wonder why I called time.sleep(). The matter is that man ysites don't allow to send a lot requests in a short period of time and we can just get 429 Error.**

# In[9]:


import time
def scrawl(category,num):
    stories = []
    for i in range(1,num+1):
        page = category+str(i)
        links = get_story_link_body(page)
        links = extract_links(links)
        stories.extend(links)
        time.sleep(0.5)
    return stories
        


# In[ ]:


tale_stories = scrawl(tale,300)


# In[ ]:


little_stories = scrawl(little,300)


# In[ ]:


print(len(tale_stories))
print(len(little_stories))


# Let's count the missed values

# In[ ]:


print(len([el for el in tale_stories if el == None]))
print(len([el for el in little_stories if el == None]))


# In[ ]:


tale_stories[0:10]


# Now let's get the texts from the links

# **Now we can extract texts from URLs! You may woner why I called time.sleep. That's because if we send a lot request in a short period of time, we will get 429 error and won't get the contents of the site.**

# In[ ]:


fantasy_texts = []
for el in fantasy_stories:
    link = 'https://americanliterature.com/short-stories-for-children/'+el
    pre_text = get_text(link)
    text = clean_text(flatten_to_string(pre_text))
    tale_texts.append(text)
    time.sleep(1)


# In[ ]:


texts_school = []
for el in school_stories:
    link = 'https://americanliterature.com/short-stories-for-children/'+el
    pre_text = get_text(link)
    text = clean_text(flatten_to_string(pre_text))
    texts_little.append(text)


# Let's count how many texts of each category we have.

# In[ ]:


print(len(texts_little))
print(len(tale_texts))
print(texts_little[20][0:1000])


# And how much of the texts are not empty? In spite of the sleep function I used while reading the articles, we still had some 429 Errors, so let's check how many articles we have really read

# In[ ]:


print(len([el for el in texts_little if len(el)>0]))
print(len([el for el in tale_texts if len(el)>0]))


# In[19]:


texts_little = [el for el in texts_little if len(el)>0]
tale_texts=[el for el in tale_texts if len(el)>0]


# In[ ]:


texts_little[10]


# In[ ]:


tale_texts[100]


# In[ ]:


texts_little = [el.replace("\r","") for el in texts_little]
tale_texts=[el.replace("\r","") for el in tale_texts]


# In[ ]:


y = ['tale' for _ in range(len(tale_texts))]
y.extend(['little' for _ in range(len(texts_little))])
print(len(y))


# In[ ]:


texts = tale_texts
texts.extend(texts_little)


# In[ ]:


df = pd.DataFrame()


# In[ ]:


df['text'] = texts
df['class'] = y


# In[ ]:


df


# In[ ]:




