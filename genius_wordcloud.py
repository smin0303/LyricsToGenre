import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud

g_rap_twc_df = pd.read_csv('genius_rap_twc.csv', index_col=0)
g_pop_twc_df= pd.read_csv('genius_pop_twc.csv', index_col=0)
g_country_twc_df = pd.read_csv('genius_country_twc.csv', index_col=0)
g_rb_twc_df = pd.read_csv('genius_rb_twc.csv', index_col=0)
g_rock_twc_df = pd.read_csv('genius_rock_twc.csv', index_col=0)

g_rap_text_freq = dict()
g_pop_text_freq = dict()
g_country_text_freq = dict()
g_rb_text_freq = dict()
g_rock_text_freq = dict()

for w, f in g_rap_twc_df.values:
    g_rap_text_freq[w] = f

for w, f in g_pop_twc_df.values:
    g_pop_text_freq[w] = f

for w, f in g_country_twc_df.values:
    g_country_text_freq[w] = f

for w, f in g_rb_twc_df.values:
    g_rb_text_freq[w] = f

for w, f in g_rock_twc_df.values:
    g_rock_text_freq[w] = f

# saved wordcloud file before I exited the window with the image to go to next wordcloud

wordcloud = WordCloud().generate_from_frequencies(g_rap_text_freq)
plt.imshow(wordcloud)
plt.axis("off")
plt.title("Word cloud for Genius rap songs")
plt.show()

wordcloud = WordCloud().generate_from_frequencies(g_pop_text_freq)
plt.imshow(wordcloud)
plt.axis("off")
plt.title("Word cloud for Genius pop songs")
plt.show()

wordcloud = WordCloud().generate_from_frequencies(g_country_text_freq)
plt.imshow(wordcloud)
plt.axis("off")
plt.title("Word cloud for Genius country songs")
plt.show()

wordcloud = WordCloud().generate_from_frequencies(g_rb_text_freq)
plt.imshow(wordcloud)
plt.axis("off")
plt.title("Word cloud for Genius R&B songs")
plt.show()

wordcloud = WordCloud().generate_from_frequencies(g_rock_text_freq)
plt.imshow(wordcloud)
plt.axis("off")
plt.title("Word cloud for Genius rock songs")
plt.show()