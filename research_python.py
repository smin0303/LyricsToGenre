import ast
import pandas as pd


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


genius = pd.read_csv('genius_first_genre.csv', index_col=0)
genius.reset_index(inplace=True, drop=True)


g_total_word_count = dict()

for index, row in genius.iterrows():
    d = ast.literal_eval(genius.loc[index, 'word freq'])
    for word in d:
        if word in g_total_word_count:
            g_total_word_count[word] += d[word]
        else:
            g_total_word_count[word] = d[word]

g_total_word_count_df = pd.DataFrame(list(g_total_word_count.items()), columns=['word', 'total count'])


g_twc_df_gte_fifty = g_total_word_count_df[g_total_word_count_df['total count'] > 49]
g_twc_df_gte_fifty = g_twc_df_gte_fifty.sort_values(by='total count', ascending=False)
g_twc_df_gte_fifty.reset_index(inplace=True, drop=True)

gte_lst = g_twc_df_gte_fifty['word'].to_list()

genius['genres'] = genius['genres'].apply(func)


words_dlst = []

for i, row in genius.iterrows():
    d = ast.literal_eval(row['word freq'])
    d_new= {key: d[key] for key in d if key in gte_lst}
    words_dlst.append(d_new)


df_lst = []

for what
wc_df_10k = pd.DataFrame.from_records(test_lst)
wc_df_10k.fillna(0, downcast='infer', inplace=True)
print(wc_df_10k.info(memory_usage="deep"))