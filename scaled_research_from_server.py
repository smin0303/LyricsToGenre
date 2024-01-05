import ast
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC, SVC, NuSVC
from sklearn.metrics import classification_report, confusion_matrix , ConfusionMatrixDisplay, accuracy_score
from sklearn import preprocessing

genius = pd.read_csv('genius_cut_again.csv', header=0, index_col=0)

twc = dict()
for index, row in genius.iterrows():
    d = ast.literal_eval(row['word freq'])
    for word in d:
        if word in twc:
            twc[word] += d[word]
        else:
            twc[word] = d[word]

twc_df = pd.DataFrame(list(twc.items()), columns=['word', 'total count'])
twc_df = twc_df.sort_values(by='total count', ascending=False).reset_index(drop=True)

n = 250
twc_df_gte_n = twc_df[twc_df['total count'] >= n]
twc_df_gte_n = twc_df_gte_n.sort_values(by='total count', ascending=False)
twc_df_gte_n.reset_index(inplace=True, drop=True)
print("Number of words with frequency equal to or greater than " + str(n) + " is " + str(twc_df_gte_n.shape[0]))

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

gte_lst = twc_df_gte_n['word'].to_list()

genius['genres'] = genius['genres'].apply(func)

print("Creating new lst")
new_lst = []

for i, row in genius.iterrows():
    for key, value in ast.literal_eval(row['word freq']).items():
        if key in gte_lst and key != "" and key != '\'':
            r = [i, row['genres']]
            r.append(key)
            r.append(value)
            new_lst.append(r)

print("making new lst DF")
lst_df = pd.DataFrame(new_lst, columns=['ID', 'genre', 'word', 'count'])
lst_df['genre'] = lst_df['genre'].astype('uint8')
intermed_df1 = pd.DataFrame(lst_df[['ID', 'genre']].groupby(['ID', 'genre']).size().reset_index(name='drop'))
y = intermed_df1.drop(['ID', 'drop'], axis=1)
y['genre'] = y['genre'].astype('uint8')
print("y made")

del intermed_df1

lst_df['genre'] = lst_df['genre'].astype('category')

lst_genre_and_lyrics = pd.pivot_table(lst_df, values='count', index='ID', columns='word', fill_value=0)
x = lst_genre_and_lyrics.reset_index(drop=True)
print("x made")

print("splitting x and y into train and test data")
train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.2,)

# new scaling technique
# only scale x-values
print("scaling training x data")
scaler = preprocessing.StandardScaler().fit(train_x)
x_scaled = scaler.transform(train_x)

print("scaling test x data")
test_scaler = preprocessing.StandardScaler().fit(test_x)
x_test_scaled = test_scaler.transform(test_x)

# linear SVC
print("Running Linear SVC")
linear = LinearSVC(random_state=0)
linear.fit(x_scaled, train_y.values.ravel())
print("Linear SVC predicting")
y_lin_predicted = linear.predict(x_test_scaled)
linear_score = accuracy_score(test_y, y_lin_predicted)
print("The score of the scaled Linear SVC is: ", linear_score)
cm_lin = confusion_matrix(test_y, y_lin_predicted)
disp_lin = ConfusionMatrixDisplay(cm_lin)
disp_lin.plot()
disp_lin.figure_.savefig("ScaledLinearSVC.png")
print("Confusion matrix of scaled linear SVC downloaded")

# SVC default - RBF
print("Running RBF SVC")
rbf = SVC(kernel='rbf', random_state=0)
rbf.fit(x_scaled, train_y.values.ravel())
print("RBF SVC predicting")
y_rbf_predicted = rbf.predict(x_test_scaled)
rbf_score = accuracy_score(test_y, y_rbf_predicted)
print("The score of the scaled RBF SVC is: ", rbf_score)
cm_rbf = confusion_matrix(test_y, y_rbf_predicted)
disp_rbf = ConfusionMatrixDisplay(cm_rbf)
disp_rbf.plot()
disp_rbf.figure_.savefig("ScaledRBFSVC.png")
print("Confusion matrix of scaled RBF SVC downloaded")

# poly SVC
print("Running poly SVC")
poly = SVC(kernel='poly', random_state=0)
poly.fit(x_scaled, train_y.values.ravel())
print("Polynomial SVC predicting")
y_poly_predicted = poly.predict(x_test_scaled)
poly_score = accuracy_score(test_y, y_poly_predicted)
print("The score of the scaled Polynomial SVC is: ", poly_score)
cm_poly = confusion_matrix(test_y, y_poly_predicted)
disp_poly = ConfusionMatrixDisplay(cm_poly)
disp_poly.plot()
disp_poly.figure_.savefig("ScaledPolynomia;SVC.png")
print("Confusion matrix of scaled polynomial SVC downloaded")

# sigmoid SVC
print("Running sigmoid SVC")
sig = SVC(kernel='sigmoid', random_state=0)
sig.fit(x_scaled, train_y.values.ravel())
print("Sigmoid SVC predicting")
y_sig_predicted = sig.predict(x_test_scaled)
sig_score = accuracy_score(test_y, y_sig_predicted)
print("The score of the scaled Sigmoid SVC is: ", sig_score)
cm_sig = confusion_matrix(test_y, y_sig_predicted)
disp_sig = ConfusionMatrixDisplay(cm_sig)
disp_sig.plot()
disp_sig.figure_.savefig("ScaledSigmoidSVC.png")
print("Confusion matrix of scaled sigmoid SVC downloaded")