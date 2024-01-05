import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud

k_rap_twc_df = pd.read_csv('kaggle_rap_twc.csv', index_col=0)
k_pop_twc_df = pd.read_csv('kaggle_pop_twc.csv', index_col=0)
k_country_twc_df = pd.read_csv('kaggle_country_twc.csv', index_col=0)
k_rb_twc_df = pd.read_csv('kaggle_rb_twc.csv', index_col=0)
k_rock_twc_df = pd.read_csv('kaggle_rock_twc.csv', index_col=0)

k_rap_text_freq = dict()
k_pop_text_freq = dict()
k_country_text_freq = dict()
k_rb_text_freq = dict()
k_rock_text_freq = dict()

for w, f in k_rap_twc_df.values:
    k_rap_text_freq[w] = f

for w, f in k_pop_twc_df.values:
    k_pop_text_freq[w] = f

for w, f in k_country_twc_df.values:
    k_country_text_freq[str(w)] = f

for w, f in k_rb_twc_df.values:
    k_rb_text_freq[w] = f

for w, f in k_rock_twc_df.values:
    k_rock_text_freq[w] = f

# saved wordcloud file before I exited the window with the image to go to next wordcloud

wordcloud = WordCloud().generate_from_frequencies(k_rap_text_freq)
plt.imshow(wordcloud)
plt.axis("off")
plt.title("Word cloud for Kaggle rap songs")
plt.show()

wordcloud = WordCloud().generate_from_frequencies(k_pop_text_freq)
plt.imshow(wordcloud)
plt.axis("off")
plt.title("Word cloud for Kaggle pop songs")
plt.show()

wordcloud = WordCloud().generate_from_frequencies(k_country_text_freq)
plt.imshow(wordcloud)
plt.axis("off")
plt.title("Word cloud for Kaggle country songs")
plt.show()

wordcloud = WordCloud().generate_from_frequencies(k_rb_text_freq)
plt.imshow(wordcloud)
plt.axis("off")
plt.title("Word cloud for Kaggle R&B songs")
plt.show()

wordcloud = WordCloud().generate_from_frequencies(k_rock_text_freq)
plt.imshow(wordcloud)
plt.axis("off")
plt.title("Word cloud for Kaggle rock songs")
plt.show()