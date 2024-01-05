import ast
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV 
from sklearn import svm


def func(g):
    if g == 'Rap':
        return 0
    elif g == 'Pop':
        return 1
    elif g == 'Country':
        return 2
    elif g == 'R&B':
        return 3
    elif g == 'Rock':
        return 4


genius = pd.read_csv('genius_cut.csv', header=0, index_col=0)       # 201,134 songs

g_total_word_count = dict()

for index, row in genius.iterrows():
    d = ast.literal_eval(genius.loc[index, 'word freq'])
    for word in d:
        if word in g_total_word_count:
            g_total_word_count[word] += d[word]
        else:
            g_total_word_count[word] = d[word]

g_total_word_count_df = pd.DataFrame(list(g_total_word_count.items()), columns=['word', 'total count'])     # takes like 1.5 min
g_total_word_count_df = g_total_word_count_df.sort_values(by='total count', ascending=False).reset_index(drop=True)

n = 100
g_twc_df_gte_n = g_total_word_count_df[g_total_word_count_df['total count'] >= n]
g_twc_df_gte_n = g_twc_df_gte_n.sort_values(by='total count', ascending=False).reset_index(inplace=True, drop=True)

gte_lst = g_twc_df_gte_n['word'].to_list()

genius['genres'] = genius['genres'].apply(func)

words_dlst = []

for i, row in genius.iterrows():
    d = ast.literal_eval(row['word freq'])
    d_new= {key: d[key] for key in d if key in gte_lst}
    words_dlst.append(d_new)