import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from statistics import mean
from sklearn.metrics import confusion_matrix
import pandas as pd
import numpy as np
import json


class DataCleaning(object):

    def __init__(self, fp, **kwargs):
        self.fp = fp
        if '.csv' in self.fp:
            self.df = pd.read_csv('data/subset.csv')
        else:
            self.df = self.read_json_file()
        self.df = self.add_needed_columns()

    def read_json_file(self):
        with open(self.fp, 'r') as json_file:
            json_list = list(json_file)
        json_list_2 = [json.loads(tweet) for tweet in json_list]
        df = pd.DataFrame(json_list_2)
        return df

    def determine_party(self, col):
        '''
        Looks through text to identify whether the user's
        description classifies them as a democrat or republican
        Input: string
        Output: string ('Dem', 'Rep', or 'unk')
        '''
        dem = ['biden', 'harris', 'democrat',
               'liberal', 'progressive', 'socialist',
               'cnn', 'msnbc']
        republican = ['trump', 'pence', 'fox',
                      'examiner', 'breitbart',
                      'maga', 'republican', 'conservative',
                      'make america great', 'keep america great']
        count = 0
        for i in dem:
            if col.lower().find(i) >= 0:
                count += 1
        for i in republican:
            if col.lower().find(i) >= 0:
                count -= 1
        if count > 0:
            return 'Dem'
        elif count == 0:
            return 'unk'
        elif count < 0:
            return 'Rep'

    def about(self, col):
        '''
        Returns 'Trump' if a string mentions Trump and
        not Biden and returns 'Biden' if a string mentions Biden and not Trump
        Input: string
        Output: string
        '''
        if col.lower().find('trump') >= 0 and col.lower().find('biden') < 0:
            return 'Trump'
        if col.lower().find('biden') >= 0 and col.lower().find('trump') < 0:
            return 'Biden'
        else:
            return 'no_candidate'

    def add_needed_columns(self):
        self.df['user_description'] = self.df.user.apply(lambda x:
                                                         x['description'])
        self.df['user_followers'] = self.df.user.apply(lambda x:
                                                       x['followers_count'])
        self.df['hastags'] = self.df.entities.apply(lambda x:
                                                    x['hashtags'])
        self.df['party'] = self.df.user_description.apply(self.determine_party)
        self.df['about'] = self.df['full_text'].apply(self.about)
        self.df['retweet'] = self.df['full_text'].apply(lambda x:
                                                        x.find('RT') >= 0)
        trump_cov_string1 = 'RT @realDonaldTrump: Tonight, @FLOTUS '
        trump_cov_string2 = 'and I tested positive for COVID-19.'
        trump_cov_string = trump_cov_string1 + trump_cov_string2
        self.df['tweet_heard_round_the_world'] = self.df['full_text'].apply(
        lambda x: x.find(trump_cov_string) >=0)
        self.no_rt = df.drop_duplicates(subset=['full_text'])
        self.df_biden = self.no_rt[self.no_rt['about'] == 'Biden']
        self.df_trump = self.no_rt[self.no_rt['about'] == 'Trump']

    def generate_sub_dfs(self):
        self.no_rt = df.drop_duplicates(subset=['full_text'])
        self.df_biden = self.no_rt[self.no_rt['about'] == 'Biden']
        self.df_trump = self.no_rt[self.no_rt['about'] == 'Trump']
        bool_ser = df_trump['tweet_hear_round_the_world'] == False
        self.df_trump_no_covid = df_trump[bool_ser]



    


if __name__ == '__main__':
    fp = 'data/concatenated_abridged.jsonl'
    data = DataCleaning(fp=fp)
    data.add_needed_columns()
    df = data.df
    print(df['user_description'])
    # df[0:1000].to_csv('data/subset.csv')
    # print(df.head())
    # print(df.info())
