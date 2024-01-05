import ast
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC, SVC, NuSVC
from sklearn.metrics import classification_report, confusion_matrix , ConfusionMatrixDisplay

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

genius = pd.read_csv('genius_cut_again.csv', header=0, index_col=0)

print("Shape of genius DF after CSV loaded: " + str(genius.shape[0]))

g_total_word_count = dict()
for index, row in genius.iterrows():
    d = ast.literal_eval(row['word freq'])
    for word in d:
        if word in g_total_word_count:
            g_total_word_count[word] += d[word]
        else:
            g_total_word_count[word] = d[word]

g_total_word_count_df = pd.DataFrame(list(g_total_word_count.items()), columns=['word', 'total count'])
g_total_word_count_df = g_total_word_count_df.sort_values(by='total count', ascending=False).reset_index(drop=True)

# words occur less than n times --> erased
# twc = total word count, gte = greater than or equal to
n = 100
g_twc_df_gte_n = g_total_word_count_df[g_total_word_count_df['total count'] >= n]
g_twc_df_gte_n = g_twc_df_gte_n.sort_values(by='total count', ascending=False)
g_twc_df_gte_n.reset_index(inplace=True, drop=True)
print("Number of words with frequency equal to or greater than " + str(n) + " is " + str(g_twc_df_gte_n.shape[0]))

gte_lst = g_twc_df_gte_n['word'].to_list()

genius['genres'] = genius['genres'].apply(func)

new_lst = []

for i, row in genius.iterrows():
    for key, value in ast.literal_eval(row['word freq']).items():
        if key in gte_lst and key != "" and key != '\'':
            r = [i, row['genres']]
            r.append(key)
            r.append(value)
            new_lst.append(r)

lst_df = pd.DataFrame(new_lst, columns=['ID', 'genre', 'word', 'count'])
lst_df['genre'] = lst_df['genre'].astype('uint8')
intermed_df1 = pd.DataFrame(lst_df[['ID', 'genre']].groupby(['ID', 'genre']).size().reset_index(name='drop'))
y = intermed_df1.drop(['ID', 'drop'], axis=1)
y['genre'] = y['genre'].astype('uint8')
print("Made y")

del intermed_df1

lst_df['genre'] = lst_df['genre'].astype('category')

print("lst_df memory usage:")
lst_df.info(memory_usage='deep')

# this should now only have 1.63 billion cells - DOES IT RUN THO???
# maybe if it runs, we should output to CSV
lst_genre_and_lyrics = pd.pivot_table(lst_df, values='count', index='ID', columns='word', fill_value=0)

# if the above runs... run the rest
x = lst_genre_and_lyrics.reset_index(drop=True)

train_features, test_features, train_labels, test_labels = train_test_split(x, y, test_size=0.2,)

# does it work on Naive Bayes?
nb_model = MultinomialNB()
nb_model.fit(train_features, train_labels.values.ravel())
score = nb_model.score(test_features, test_labels)
print(score)

y_predicted = nb_model.predict(train_features)

cm = confusion_matrix(test_labels, y_predicted)
disp = ConfusionMatrixDisplay(cm)
disp.plot()