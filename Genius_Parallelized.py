import requests
import pandas as pd
from bs4 import BeautifulSoup
import re
import multiprocessing
import collections
import time

# make api_df
columns = ['title', 'artists', 'lyrics', 'word freq', 'unique word count', 'total word count', 'genres']
api_df = pd.DataFrame(columns=columns)

access_token = "access_token_from_genius_api"

token = 'Bearer {}'.format(access_token)
headers = {'Authorization': token}

def get_df(id):
    try:
        r = requests.get('https://api.genius.com/songs/' + str(id), headers=headers)

        if r.status_code != 200:
            return
        
        song = r.json()["response"]["song"]
        
        if song["language"] != 'en':
            return

        if song["lyrics_state"] != "complete":
            return
        
        url = song["url"]

        url_request = requests.get(url)

        bs = BeautifulSoup(url_request.text.replace('<br/>', '\n'), "html.parser")
        div = bs.find("div", class_=re.compile("^lyrics$|Lyrics__Root"))
        tags = bs.find_all("a", class_=re.compile("SongTags__Tag"))

        if (div is None) or (len(tags) == 0):
            return

        genres = [tag.get_text() for tag in tags]
        if 'Non-Music' in genres:
            return
        
        # cleaning lyrics 
        # get rid of lines with "Verse", "Chorus", band member name, etc
        # erase any special characters except ' and replace new lines and/or multiple spaces with one space
        # use all lowercase letters
        lyrics = div.get_text()
        lyrics = " ".join(lyrics.split("\n")[:])
        lyrics = lyrics[:-5]
        lyrics = re.sub(r'((\[.*?\])*)|([^A-z\s\'])', '', lyrics)
        lyrics = re.sub(r'[ ]{2,}', ' ', lyrics)
        lyrics = lyrics.lower()

        word_lst = lyrics.split(" ")
        word_freq = dict(collections.Counter(word_lst))

        api_df.loc[id] = song["title"], song["artist_names"], lyrics, word_freq, len(word_freq), len(word_lst), genres
    except:
        print("Song information for ID: " + str(id) + " not found\n")
    return api_df


if __name__ == '__main__':
    # skip 200001 - 400000
    # skip 600001 - 800000 
    # skip 1000001 - 1200000?
    i = 120
    stop = 125
    pool = multiprocessing.Pool()
    while i < stop:
        results = pool.map(get_df, range((i * 1000) + 1,  ((i + 1) * 1000) + 1))
        combined = pd.concat(results)
        final = combined.drop_duplicates(subset=['title', 'artists', 'lyrics', 'unique word count', 'total word count'])
        final.to_csv('final.csv', mode='a', index=False, header=False)
        print("Done. Added IDs " + str((i * 1000) + 1) + " to " + str((i + 1) * 1000) + " to CSV")
        time.sleep(200)
        i += 1
    pool.close()