from importlib.metadata import metadata
from urllib import request
import flask
import pandas as pd
import numpy as np

from flask import Flask ,render_template,request
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

cv=CountVectorizer()

def get_cosine_similarity(data):
    x=data['features']
    vectors=cv.fit_transform(x)
    cosine=cosine_similarity(vectors)
    return pd.DataFrame(cosine)


def recommend(id,features,similarity):
    a=list(features['id'])
    a=a.index(id)
    s=list(enumerate(similarity[a]))
    s=sorted(s,key=lambda x: x[1], reverse=True)
    s=s[1:9]
    index=[add[0] for add in s]
    index=[features['id'][x] for x in index]
    return index

def is_there(title,movies):
    a=list(movies['title'].str.lower())
    title=title.lower()
    if title in a:
        loc=a.index(title)
        return loc
    else:
        return -1



app = Flask(__name__)
@app.route('/')

def hello():
    return render_template('home.html')

@app.route('/',methods=['GET','POST'])
def main():
    input=request.form['name']
    metadata=pd.read_csv(r'C:\Users\pande\OneDrive\Documents\Movie-Recommender\Datasets\final_metadata2.csv',encoding='latin-1')
    features=pd.read_csv(r'C:\Users\pande\OneDrive\Documents\Movie-Recommender\Datasets\features2.csv',encoding='latin-1')
    cosine_similarity=get_cosine_similarity(features)

    loc=is_there(input,metadata)
    if loc==-1:
        return render_template('not_found.html')
    id=metadata['id'][loc]
    rcmd=recommend(id,features,cosine_similarity)
    return render_template('details.html',id=id,movies=rcmd)




if __name__=='__main__':
  app.run(debug=True)