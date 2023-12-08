import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from urllib.request import urlopen
import json
from copy import deepcopy
from textblob import TextBlob
from wordcloud import WordCloud
from bs4 import BeautifulSoup
from collections import Counter
import seaborn as sns
import numpy as np
from ast import literal_eval
from collections import Counter
from plotly.subplots import make_subplots

@st.cache_data
def load_data(path):
    df = pd.read_csv(path)
    return df

mastodon_final_nodup = load_data(path="./data/mastodon_final_nodup.csv")
mastodon_final = deepcopy(mastodon_final_nodup)

st.title('Mastodon Analysis')

st.header('Overview of Mastodon')


option = st.selectbox('Select an option:', ['Why Does This Matter', 'What Is Mastodon', 'How Did We Fetch And Analyse Data', 'References'])


if option == 'Why Does This Matter':
    st.write('Market Shift in the wake of Elon Musk\'s acquisition of Twitter, now X. \
             Mastodon\'s Opportunity in response to this shift through its increased activity and user engagement.\
            Understanding the influence of X\'s  trends on Mastodon,\
            indicating the platform\'s potential to attract X\'s dissatisfied users.')

elif option == 'What Is Mastodon':
    st.write('Unlike Twitter\'s single platform, Mastodon is a network of many independent servers.')
    st.write('Each server has its own rules, moderated by its community, not a central authority.')
    st.write('Different Mastodon instances can interact, similar to email providers-')

elif option == 'How Did We Fetch And Analyse Data':
    st.write('Data Selection: Sample: top trending hashtags on X in November 2023')
    st.write('Mastodon Data Fetch: Searched Mastodon.social instance for Toots with these hashtags,')
    st.write('yielding over 30,000 results.')
    st.write('Objective: Identify Mastodon\'s global expansion potential and its response')
    st.write('to Twitter\'s management changes; and basically explore the data found for further insights')
    
elif option == 'References':
    st.write('https://www.ohchr.org/en/press-releases/2023/11/gaza-un-experts-call-international-community-prevent-genocide-against')
    st.write('https://www.dw.com/en/sudan-conflict-eu-warns-of-another-genocide-in-darfur/a-67381833')
    st.write('https://www.nytimes.com/2023/12/04/business/spotify-layoffs.html')
    st.write('https://www.bbc.com/news/av/world-latin-america-67473529')
    st.write('https://www.forbes.com/sites/garydrenik/2023/09/08/unveiling-x-the-implications-of-twitters-bold-rebranding-move/?sh=38c0cc0c2ff2')
    st.write('https://www.itworldcanada.com/article/how-mastodon-may-succeed-as-the-twitter-alternative-where-others-have-failed/514395')
    st.write('https://www.twitter-trending.com/worldwide/statistics')
    st.write('https://jrashford.com/2023/02/13/how-to-scrape-mastodon-timelines-using-python-and-pandas/')


st.header('User Growth Data')

if option == 'All':  # Corrected condition
    alt_mastodon_final = mastodon_final
else:
    alt_mastodon_final = mastodon_final[mastodon_final["tags"].str.contains(option, case=False, na=False)]

# Matplotlib Map
st.subheader("MatplotLib Map")

mastodon_final['account.created_at'] = pd.to_datetime(mastodon_final['account.created_at'], utc=True)
mastodon_final['account_age'] = (pd.to_datetime('today', utc=True) - mastodon_final['account.created_at']).dt.days

plt.figure(figsize=(10, 6))
for hashtag in ['argentina', 'gaza', 'israel', 'milei', 'missuniverse', 'spotify', 'thanksgiving', 'trump', 'twitter', 'ons']:
    hashtag_df = mastodon_final[mastodon_final['tags'].str.contains(hashtag, case=False, na=False)]
    sns.histplot(hashtag_df['account_age'], color='grey', kde=False, alpha=0.3)

plt.title('Distribution of Mastodon Account Ages for Selected Hashtags Before/After Twitter Acquisition')
plt.xlabel('Account Age (Days Since Creation)')
plt.ylabel('Number of New User Accounts')
plt.xlim(0,1000)
plt.axvline(400, color='red',linewidth=3)

st.pyplot(plt)

# Plotly Map
st.subheader("Plotly Map")

mastodon_final['account.created_at'] = pd.to_datetime(mastodon_final['account.created_at'], utc=True)

mastodon_final['account_age'] = (pd.to_datetime('today', utc=True) - mastodon_final['account.created_at']).dt.days

fig = go.Figure()

for hashtag in ['argentina', 'gaza', 'israel', 'milei', 'missuniverse', 'spotify', 'thanksgiving', 'trump', 'twitter', 'ons']:
    hashtag_df = mastodon_final[mastodon_final['tags'].str.contains(hashtag, case=False, na=False)]
    fig.add_trace(go.Histogram(
        x=hashtag_df['account_age'],
        opacity=0.7,
        name=f'#{hashtag}',
        histnorm='probability',
    ))

fig.add_shape(
    type='line',
    x0=400,
    x1=400,
    y0=0,
    y1=1,
    line=dict(color='black', width=2, dash='dash'),
)

fig.update_layout(
    title='Account Age Distribution for Hashtags',
    xaxis_title='Account Age (Days)',
    yaxis_title='Probability',
    barmode='overlay',
)
st.plotly_chart(fig)

st.header('Trending Topics')

# Matplotlib Map
st.subheader("MatplotLib Map")

hashtags = mastodon_final['tags'].apply(lambda x: [] if pd.isna(x) else [tag['name'] for tag in literal_eval(x)])
all_hashtags = [hashtag for hashtags_list in hashtags for hashtag in hashtags_list]
hashtag_counts = Counter(all_hashtags)
most_common_hashtags = hashtag_counts.most_common(20)

most_common_hashtags_twitter = [('argentina', 4.60 * 1e6),
                                ('gaza', 5.92  * 1e6),
                                ('israel', 5.80  * 1e6),
                                ('milei', 4.98  * 1e6),
                                ('ons', 3.96  * 1e6),
                                ('spotify', 4.82  * 1e6),
                                ('thanksgiving', 4.98  * 1e6),
                                ('MissUniverseThailand2023', 4.08 * 1e6),
                                ] # 'trump': , 'twitter':  # missing volums


most_common_hashtags_mastodon = most_common_hashtags

most_common_hashtags_twitter_tag = [tag[0] for tag in most_common_hashtags_twitter]
most_common_hashtags_twitter_count = [np.log(tag[1]) for tag in most_common_hashtags_twitter]

# extracting tags and count of them_mastodon
most_common_hashtags_mastodon_tag = [tag[0] for tag in most_common_hashtags_mastodon]
most_common_hashtags_mastodon_count = [np.log(tag[1]) for tag in most_common_hashtags_mastodon]
tags_combined = list(set(most_common_hashtags_twitter_tag + most_common_hashtags_mastodon_tag))
count_mastodon = []
count_twitter = []
for tag in tags_combined:
        try:
            count = most_common_hashtags_mastodon_count[most_common_hashtags_mastodon_tag.index(tag)]
        except:
            count = 0
        count_mastodon.append(count)

        try:
            count = most_common_hashtags_twitter_count[most_common_hashtags_twitter_tag.index(tag)]
        except:
            count = 0
        count_twitter.append(count)

combined_data = list(zip(tags_combined, count_mastodon, count_twitter))

# Sort the list of tuples based on the count_twitter values (descending order)
sorted_data = sorted(combined_data, key=lambda x: x[2], reverse=True)

# Unzip the sorted data back into separate arrays
sorted_tags_combined, sorted_count_mastodon, sorted_count_twitter = zip(*sorted_data)

plt.figure()
plt.bar(x=sorted_tags_combined, height=sorted_count_mastodon, color='purple', alpha=.75)
plt.bar(x=sorted_tags_combined, height=sorted_count_twitter, alpha=.75)
plt.xticks(rotation=90)
plt.legend(['mastodon','twitter'])
font_dict = {'fontsize': 14, 'fontweight': 'bold', 'fontfamily': 'serif'}
plt.xlabel('Hashtag', fontdict=font_dict)
plt.ylabel('Count (log)', fontdict=font_dict)
plt.title('Comparison of Most popular Hashtags \namongst Mastodon and Twitter Counts', fontsize= 14, fontweight= 'bold')

st.pyplot(plt)

# Plotly Map
st.subheader("Plotly Map")

tag_of_interest = 'milei'
df = mastodon_final[mastodon_final['tags'].str.contains(tag_of_interest, case=False, na=False)]

# Convert 'created_at' to datetime
df['created_at'] = pd.to_datetime(df['created_at'])

# Resample to get daily counts
daily_counts = df.resample('D', on='created_at').count()

# Create Plotly figure
fig = go.Figure()

# Add scatter (line plot) to the figure
fig.add_trace(go.Scatter(x=daily_counts.index, 
                         y=daily_counts['id'], 
                         mode='lines+markers', 
                         name='posts'))

# Update title and axis 
fig.update_layout(
    title='Daily Posts Trend for Hashtag #milei',
    xaxis_title='Date',
    yaxis_title='Number of Posts',
    xaxis=dict(tickangle=45),
    template="plotly_white"  
)

st.plotly_chart(fig)

st.header('Language Distribution')

# Matplotlib Map
st.subheader("MatplotLib Map")

language_codes = {
    'en': 'English', 
    'es': 'Spanish', 
    'it': 'Italian', 
    'ro': 'Romanian',
    'ja': 'Japanese',
    'ca': 'Cantonese',
    'fr': 'French',
    'pt': 'Portuguese',
    'de': 'Deutsch',
    'nl': 'Dutch',
}

language_distribution = df['language'].value_counts().head(10)
language_distribution.index = language_distribution.index.map(lambda code: language_codes.get(code, 'Unknown').capitalize())
plt.figure(figsize=(10, 6))

language_distribution.plot(kind='bar')
plt.title('Number of Posts in Specific Languages - Top 10')
plt.xlabel('Language')
plt.ylabel('Number of Posts')

plt.xticks(rotation=45)

plt.grid(axis='y')

plt.tight_layout()

st.pyplot(plt)

# Plotly Map
st.subheader("Plotly Map")

language_full_names = {
    'en': 'English', 
    'es': 'Spanish',
    'de': 'German',
    'fr': 'French',
    'it': 'Italian',
    'nl': 'Dutch',
    'pt': 'Portuguese',
    'pt-br': 'Portuguese (Brazil)'
}

tag_of_interest = 'milei'
df = mastodon_final[mastodon_final['tags'].str.contains(tag_of_interest, case=False, na=False)]
desired_languages = ['en', 'es', 'de', 'fr', 'it', 'nl', 'pt', 'pt-br']
language_distribution = df[df['language'].isin(desired_languages)]['language'].value_counts()

fig = go.Figure()

fig.add_trace(
    go.Bar(x=language_distribution.index.map(language_full_names), y=language_distribution.values, name=f"Language Distribution for '{tag_of_interest}'")
)

fig.update_layout(
    title_text=f"Language Distribution for '{tag_of_interest}'",
    xaxis_title="Language",
    yaxis_title="Number of Posts",
    height=600,
    width=800
)

st.plotly_chart(fig)

st.header('Sentiment Analysis')

# Plotly Map I
st.subheader("Plotly Map I")

df_sentiments=pd.read_csv('./data/mastodon_sentiment.csv')

sentiment_counts = df_sentiments['sentiment'].value_counts()

fig = go.Figure([go.Bar(x=sentiment_counts.index, y=sentiment_counts.values)])

fig.update_layout(
    title='Sentiment Distribution of Mastodon Posts',
    xaxis_title='Sentiment Category',
    yaxis_title='Number of Posts'
)

st.plotly_chart(fig)

# Plotly Map II
st.subheader("Plotly Map II")

fig = go.Figure(data=[go.Histogram(x=df_sentiments['sentiment_score'], nbinsx=50,)])

# Update layout 
fig.update_layout(
    title_text='Distribution of Sentiment Scores in Selected Hashtags', 
    xaxis_title_text='Sentiment Score', 
    yaxis_title_text='Count', 
    bargap=0.2, 
    bargroupgap=0.1, # gap between bars of the same location coordinates
    height=800
)

st.plotly_chart(fig)