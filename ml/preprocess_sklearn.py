# -*- coding: utf-8 -*-
"""
Created on Wed Jul  7 14:34:19 2021

@author: Lenovo
"""
from zipfile import ZipFile
import os
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import MultiLabelBinarizer
import joblib
from io import StringIO, BytesIO
import json
import argparse

z_file = r'C:\Users\Lenovo\Downloads\imdbTop250.csv.zip'
csv_file = 'imdbTop250.csv'

'''
https://www.kaggle.com/mustafacicek/imdb-top-250-lists-1996-2020
Index(['Ranking', 'IMDByear', 'IMDBlink', 'Title', 'Date', 'RunTime', 'Genre',
       'Rating', 'Score', 'Votes', 'Gross', 'Director', 'Cast1', 'Cast2',
       'Cast3', 'Cast4'],
      dtype='object')
'''

def get_df(input_path,remove_duplicates='False'):
    zip_path = input_path[:input_path.rfind('\\')]
    csv_file = input_path[len(zip_path)+1:]
    with ZipFile(zip_path, 'r') as z:
        # printing all the contents of the zip file
        df = pd.read_csv(z.open(csv_file)) 
    if remove_duplicates:
        df = df.drop_duplicates('Title').reset_index(drop=True)
        
    columns = list(df.columns)
    i = columns.index('Gross')
    columns = [columns[i]]+columns[:i]+columns[i+1:]
    df = df[columns]
    return df
    
def input_fn(input_path, content_type='zip'):
    if content_type == 'text/csv':
        # Read the raw input data as CSV.
        df = pd.read_csv(StringIO(input_path), 
                         header=None)
    else:
       df = get_df(input_path)

    df[df.columns[12:]] = df[df.columns[12:]].fillna('NaN').applymap(lambda x: x.strip())
    cast = df[df.columns[12:]].values
    cast = cast.reshape(-1,).astype('U')
    cast = pd.Series(cast)
    cast_vc = cast.value_counts()
    cast_vc['NaN'] = 0
    m = cast_vc.max()
    df['Cast'] = df[df.columns[12:]].applymap(dict(cast_vc).get).sum(axis=1)/m
    df = df.drop(columns=['Title','Ranking','IMDByear','IMDBlink','Cast1','Cast2','Cast3','Cast4'])
    
    director_vc = df['Director'].value_counts()
    m = director_vc.max()
    df['Director'] = df['Director'].apply(dict(director_vc).get)/m
    
    #mean = (df['Score']/df['Rating']).dropna().mean()
    #nanindex = df['Score'][df['Score'].apply(np.isnan)].index
    #df['Score'].iloc[nanindex] = df['Rating'].iloc[nanindex]*mean
    return df

def save_model(model,model_dir):
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)
    joblib.dump(model, os.path.join(model_dir,"model.joblib"))
    
def model_fn(model_dir):
    return joblib.load(os.path.join(model_dir, "model.joblib"))

def genre_process(df):
    mlb = MultiLabelBinarizer()
    genre = df['Genre'].apply(lambda x: x.replace(' ','')).values
    l = [i.split(',') for i in genre]
    garray = mlb.fit_transform(l)
    genre_list = mlb.classes_.astype('U')
    
    return pd.DataFrame(garray,columns=genre_list)

def process(input_data):
    df = input_data
    garray = genre_process(df)
    df = df.drop(columns=['Genre'])
    
    #red_garray = pd.DataFrame(model.fit_transform(garray),columns=[f'Genre{i}' for i in range(n_components)])
    #save_model(model,model_dir)    
    
    return pd.concat([df,garray],axis=1)

def train(input_data,model,model_dir):
    garray = genre_process(input_data)

    red_garray = pd.DataFrame(model.fit_transform(garray),columns=[f'Genre{i}' for i in range(n_components)])
    save_model(model,model_dir)    

def save_model(model,model_dir):
    joblib.dump(model, os.path.join(model_dir,'pcamodel.sav'))
    
def predict_fn(input_data, model):
    garray = genre_process(input_data)
    df = input_data
    red_garray = pd.DataFrame(model.transform(garray),columns=[f'Genre{i}' for i in range(n_components)])
    
    df = df.drop(columns=['Genre'])
    df = pd.concat([df,red_garray],axis=1).values
    
    return df

def output_fn(prediction, accept="application/json"):
    """Format prediction output

    The default accept/content-type between containers for serial inference is JSON.
    We also want to set the ContentType or mimetype as the same value as accept so the next
    container can read the response payload correctly.
    """
    if accept == "application/json":
        instances = []
        for row in prediction.tolist():
            instances.append({"features": row})

        json_output = {"instances": instances}
        return json.dumps(json_output)
    '''
        #return worker.Response(json.dumps(json_output), accept, mimetype=accept)
        
    elif accept == 'text/csv':
        return worker.Response(encoders.encode(prediction, accept), accept, mimetype=accept)
    else:
        raise RuntimeException("{} accept type is not supported by this script.".format(accept))
    '''
    return

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-data-dir', type=str, default=os.environ['SM_OUTPUT_DATA_DIR'])
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--train', type=str, default=os.environ['SM_CHANNEL_TRAIN'])

    args = parser.parse_args()

    input_data = input_fn(args.train,args.model_dir)
    n_components = 16
    pca = PCA(n_components=n_components)
    train(input_data,pca,args.model_dir)
    

#-----------------

import re
df2 = pd.read_csv(r'C:\Users\Lenovo\Downloads\Attributes_DataFrame.csv')
'''
https://www.kaggle.com/lukelarue/movie-attributes-for-3400-movies-from-20002020
'Title', 'Domestic', 'International', 'Budget', 'Distributor',
       'MPAA-Rating', 'Runtime', 'Genres'
'''
pattern = '\(\d+\)'
c = re.compile(pattern)
df2['Year'] = df2['Title'].apply(lambda x: int(c.search('The Wolf of Snow Hollow (2020)').group()[1:-1]))
df2['Title'] = df2['Title'].apply(lambda x:x.split('(')[0].strip())


z_file = r'C:\Users\Lenovo\Downloads\IMDb movies.csv.zip'
csv_file = 'IMDb movies.csv'
with ZipFile(z_file, 'r') as z:
    # printing all the contents of the zip file
    df3 = pd.read_csv(z.open(csv_file))
    
'''
'title', 'original_title', 'year', 'genre', 'duration', 'director',
       'writer', 'production_company', 'actors', 'description', 'avg_vote',
       'votes', 'budget', 'usa_gross_income', 'worlwide_gross_income',
       'metascore', 'reviews_from_users', 'reviews_from_critics'
'''
df3['year'] = df3['year'].replace('TV Movie 2019',2019).astype('int')
df3 = df3[df3['year']>=2000].reset_index()
df3 = df3.drop(columns=['imdb_title_id','date_published','country','language'])

