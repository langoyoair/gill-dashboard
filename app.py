import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from nltk.stem import *
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
import re
import nltk
from itables import show  
import numpy as np
import tqdm
import gensim
from gensim.utils import simple_preprocess
import gensim.corpora as corpora
import pyLDAvis.gensim
import pickle 
import pyLDAvis
import os
import streamlit as st
from wordcloud import WordCloud
from stoc import stoc




# STREAMLIT CONFIG
st.set_page_config(layout="wide")






@st.cache_data()
def load_datasets():
    posts = pd.read_csv('posts_reduced.csv',sep='|')
    comments = pd.read_csv('comments_reduced.csv',sep='|')
    communities = pd.read_csv('reddit_communities.csv',sep=',')
    # Setting the dates as datetimes
    posts['date'] = pd.to_datetime(posts['date'], format='mixed', errors='coerce')
    comments['date'] = pd.to_datetime(comments['date'], format='mixed', errors='coerce')
    return posts, comments, communities


def plot_history(df,category='lgbt'):
    new_index =  df.loc[category].index.map(lambda x: '-'.join(map(str, x)))
    fig = go.Figure([go.Bar(x=new_index, y=df.loc[category]['score'])])
    fig.update_layout(height=1000, width=1000)
    st.plotly_chart(fig)


def plot_category_counts(df):
    count_category_df = df['category'].value_counts().reset_index()

    fig = px.bar(count_category_df, x='category', y="count", color='category')
    st.plotly_chart(fig)

def print_top_comments(subreddit):
    """
    Function that displays top comments for a subreddit
    """
    top_comments = pd.merge(left=comments[['post_id','message', 'score']], right=posts[['post_id','text','category','url']], on="post_id", how="left")

    top_comments = pd.merge(left=comments[['post_id','message', 'score']], right=posts[['post_id','text','category','url']], on="post_id", how="left")
    top_comments = top_comments.sort_values(by='score', ascending=False)
    top_comments = top_comments[top_comments['category'] == subreddit]
    top_comments = top_comments.drop_duplicates()
    # pd.set_option('display.max_colwidth', None)

    top_comments.head(10)[['text','message','score','url']]

def process_text(raw_text):
    #Consideramos únicamente letras utilizando una expresión regular
    letters_only = re.sub("[^a-zA-Z]", " ",raw_text) 
    #Convertimos todo a minúsculas
    words = letters_only.lower().split()
    
    #Eliminamos las stopwords
    stops = set(stopwords.words("english")) 
    not_stop_words = [w for w in words if not w in stops]
    
    # #Lematización
    # wordnet_lemmatizer = WordNetLemmatizer()
    # lemmatized = [wordnet_lemmatizer.lemmatize(word) for word in not_stop_words]
    
    # #Stemming
    # stemmer = PorterStemmer()
    # stemmed = [stemmer.stem(word) for word in lemmatized]
    
    return( " ".join( not_stop_words ))  




def plot_sentiments(df):
    from plotly.subplots import make_subplots
    import plotly.graph_objects as go

    sentiment_by_comunnity = df.groupby([(df["category"])]).agg(
    {'score':'mean',
    'positive':'mean', 
    'negative':'mean', 
    'neutral':'mean', 
    'hateful':'mean', 
    'aggressive':'mean',
    'joy':'mean', 
    'sadness':'mean', 
    'surprise':'mean', 
    'others':'mean', 
    'disgust':'mean', 
    'fear':'mean', 
    'anger':'mean',
    }
)

    sentiment_by_comunnity = sentiment_by_comunnity.round(4)

    fig = make_subplots(rows=4, cols=3, )

    x = 0
    for i in range(1,5):
        for j in range(1,4):
            x+=1
            sent = sentiment_by_comunnity.columns[x]
            fig.add_trace(
                go.Bar( y=sentiment_by_comunnity.index, x=sentiment_by_comunnity[sent], text=sentiment_by_comunnity[sent], name=sent, orientation='h'),
                row=i, col=j
            )


    fig.update_layout(height=1000, width=1000)
    st.plotly_chart(fig)

def top_k_ngrams(df, K=50, N_1=1, N_2=1):
    """
    Plots a histogram with the top k N-grams

    df :param: column from datafre to analyze
    k :param: top k n-grams
    N_1: min n-gram to consider
    N_2: max n-gram to consider  
    """

    # df['clean_text'] = df.apply(lambda x: process_text(str(x)))

    df['clean_text'] = df['clean_text'].fillna('')
    word_vectorizer = CountVectorizer(ngram_range=(N_1, N_2), analyzer='word', decode_error='ignore' )
    sparse_matrix = word_vectorizer.fit_transform(df['clean_text'])
    vocab = word_vectorizer.get_feature_names_out()
    freq = sparse_matrix.sum(axis=0).A1
    ngrams_df = pd.DataFrame({'ngram': vocab, 'frequency': freq})

    # Ordenar el DataFrame por frecuencia en orden descendente
    ngrams_df = ngrams_df.sort_values(by='frequency', ascending=False)

    #Remove certain ngrams that don't add value
    words_to_drop = ['reddit', 'http', 'www', 'https', 'youtube', 'wiki', 'co', 'org', 'com', 'bot', 'gov', 'nlm', 'nih']
    ngrams_df = ngrams_df[~ngrams_df['ngram'].str.contains('|'.join(words_to_drop))]

    # Mostrar los primeros K n-gramas más frecuentes
    top_ngrams = ngrams_df.head(K)
    top_ngrams = top_ngrams.set_axis(top_ngrams['ngram'])


    


    # Create trace
    trace = go.Bar(
        x=top_ngrams['frequency'],
        y=top_ngrams['ngram'],
        orientation='h',
        marker=dict(color='skyblue')
    )

    # Create layout
    layout = go.Layout(
        title=f'Top {K} more frequent N-Grams',
        yaxis=dict(title='Frequency'),
        xaxis=dict(title='N-Gram'),
        margin=dict(l=150),  # Adjust left margin for longer y-axis labels
    )

    # Create figure
    fig = go.Figure(data=[trace], layout=layout, )
    fig.update_layout(height=1000, width=1000)

    # Show figure
    st.plotly_chart(fig)


def ngram_wordcloud(df, K=50, N_1=1, N_2=1):
    """
    Plots a wordcloud with the top k N-grams

    df :param: column from datafre to analyze
    k :param: top k n-grams
    N_1: min n-gram to consider
    N_2: max n-gram to consider  
    """
    df['clean_text'] = df['clean_text'].fillna('')
    word_vectorizer = CountVectorizer(ngram_range=(N_1, N_2), analyzer='word')
    sparse_matrix = word_vectorizer.fit_transform(df['clean_text'])
    vocab = word_vectorizer.get_feature_names_out()
    freq = sparse_matrix.sum(axis=0).A1
    ngrams_df = pd.DataFrame({'ngram': vocab, 'frequency': freq})

    # Ordenar el DataFrame por frecuencia en orden descendente
    ngrams_df = ngrams_df.sort_values(by='frequency', ascending=False)

    #Remove certain ngrams that don't add value
    words_to_drop = ['reddit', 'http', 'www', 'https', 'youtube', 'wiki', 'co', 'org', 'com', 'bot', 'gov', 'nlm', 'nih']
    ngrams_df = ngrams_df[~ngrams_df['ngram'].str.contains('|'.join(words_to_drop))]

    # Mostrar los primeros K n-gramas más frecuentes
    top_ngrams = ngrams_df.head(K)
    top_ngrams = top_ngrams.set_axis(top_ngrams['ngram'])

    # Generate a word cloud image
    wordcloud = WordCloud(background_color='white',width=2000, height=2000)
    wordcloud.generate_from_frequencies(frequencies=top_ngrams['frequency'])
    plt.figure(figsize=(10, 10))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    # plt.show()
    st.pyplot(plt)



def display_table(subreddits, keywords):
    top_comments = pd.merge(left=comments[['post_id','message', 'score']], right=posts[['post_id','text','category','url']], on="post_id", how="left")
    top_comments = top_comments.sort_values(by='score', ascending=False)
    top_comments = top_comments[top_comments['category'].isin(subreddits)]
    try:
        top_comments = top_comments[top_comments['message'].str.contains('|'.join(keywords))]
    except:
        pass
    top_comments = top_comments.drop_duplicates()
    # pd.set_option('display.max_colwidth', None)

    st.dataframe(top_comments)



def LDA(texts):
    # def preprocess_LDA(raw_text):
    #     #Consideramos únicamente letras utilizando una expresión regular
    #     letters_only = re.sub("[^a-zA-Z]", " ",raw_text) 
    #     #Convertimos todo a minúsculas
    #     words = letters_only.lower().split()
        
    #     #Eliminamos las stopwords
    #     stops = set(stopwords.words("english")) 
    #     not_stop_words = [w for w in words if not w in stops]
    #     return not_stop_words
    # preprocessed_texts = [preprocess_LDA(str(text)) for text in texts]
    texts = texts.dropna()
    preprocessed_texts =  texts.str.split(' ')
    # Create Dictionary
    id2word = corpora.Dictionary(preprocessed_texts)
    # Term Document Frequency
    corpus = [id2word.doc2bow(text) for text in preprocessed_texts]
    # View
    # number of topics
    num_topics = 2
    # Build LDA model
    lda_model = gensim.models.LdaMulticore(corpus=corpus,
                                       id2word=id2word,
                                       num_topics=num_topics)
    # Visualize the topics
    pyLDAvis.enable_notebook()
    LDAvis_prepared = pyLDAvis.gensim.prepare(lda_model, corpus, id2word)
    LDAvis_data_filepath = os.path.join('./results/ldavis_prepared_'+str(num_topics))
    # this is a bit time consuming - make the if statement True
    # if you want to execute visualization prep yourself
    if 1 == 1:
        LDAvis_prepared = pyLDAvis.gensim.prepare(lda_model, corpus, id2word)
        with open(LDAvis_data_filepath, 'wb') as f:
            pickle.dump(LDAvis_prepared, f)
     #load the pre-prepared pyLDAvis data from disk
    with open(LDAvis_data_filepath, 'rb') as f:
        LDAvis_prepared = pickle.load(f)
    pyLDAvis.save_html(LDAvis_prepared, './results/ldavis_prepared_'+ str(num_topics) +'.html')
    return LDAvis_prepared
    

if __name__ == '__main__':
        
    posts, comments, communities = load_datasets()

    toc = stoc()

    toc.h1("Reddit analysis")
    st.write("This is an analysis on equality and gender based subreddits. Reddit is nowadays a well known platform for discussions. As a consequence there are plenty of conversations on gender topics. Reddit’s advantage is that the platform is subdivided into self-organized forums called “subreddits”. Each of these communities is normally dedicated to the discussion of one topic or one view of said topic. In addition, its voting feature allows querying each community in order to identify which are the posts that have more popularity and visibility. These features permit an extraction based on selected communities and a rapid access to the most popular opinions.")
    toc.h2("Datasets")
    st.write("These are the datasets involved in this study. Extracted using [Reddit API](https://old.reddit.com/dev/api/).")
    toc.h3("Communities")
    st.write("These are the communities involve in this study with their descriptions. Several subreddits were selected in order to percieve different views on gender and equality: from the most adanvced and supportive groups to some of them including hate speech.")
    st.dataframe(communities[['Name', 'Description', 'Source', "Members (April 24)"]])
    toc.h3("Posts")
    st.write("This is a dataset that contains the posts extracted from reddit. It involves the top 1000 posts all time from each of them.")
    st.dataframe(posts[["title","message","date","category","author","url","score","upvote_ratio","source","n_comments","post_id"]])
    toc.h3("Comments")
    st.write("These are a sample the comments that are replies to each of the posts in the previous dataset. More than a million comments were collected for this.")

    st.dataframe(comments[['id','author','message','date','post_id','score','category']].head(500000))

    # These are the possible subreddits
    categories = posts.category.unique()

    # Aggregating the number of posts by year-month
    posts_date_agg = posts.groupby([(posts["category"]),(posts['date'].dt.year), (posts['date'].dt.month)]).count()
    comments_date_agg = comments.groupby([(comments["category"]), (comments['date'].dt.year), (comments['date'].dt.month)]).count()

    posts_date_agg.index.map(lambda x: '-'.join(map(str, x)))



    toc.h2("Members by subreddit")
    st.write("Here the number of members in each subreddit can be appreciated. The number of members were recorded on April 2024 although the extractions were finished earlier. Note that \"antifeminists\" is now banned so it displays 0 but its data was extracted when it still was public. LGBT is clearly the most popular subreddit. Closely follow Feminism, AskFeminsts, NonBinary, trans or gay. MensRIghts and MensLibe also seem popular. Other subreddits are less populated.")

    members = communities[['Name', 'Members (April 24)']]
    fig = px.bar(members, x='Members (April 24)', y='Name', title='Memebers per community', text='Members (April 24)'  , color='Name',color_continuous_scale=px.colors.sequential.Inferno,  height=750, width= 750)
    st.plotly_chart(fig)

    

    # REDDIT COMMUNITIES SCORES
    toc.h2("Score distribution")
    st.write("Posts and comments have a voiting score system and they can be upvoted or downvoted. Here is how each of the subreddits perform according to their scores.")

    post_scores = posts.groupby([(posts["category"])]).agg(
        {'score':['mean', 'std', 'min', 'max'],
        }
    ).round(2)

    fig = px.box(posts, x="score", y="category", color="category", height=1000, width= 1000)

    st.plotly_chart(fig)


    comment_scores = comments.groupby([(comments["category"])]).agg(
        {'score':['mean', 'std', 'min', 'max'],
        }
    ).round(2)

    toc.h3("Posts")
    fig = make_subplots(rows=2, cols=2)

    fig.add_trace(
        go.Bar( x=post_scores.index, y=post_scores["score"]['mean'], text=post_scores["score"]["mean"], name="Mean"),
        row=1, col=1
    )


    fig.add_trace(
        go.Bar( x=post_scores.index, y=post_scores["score"]['std'], text=post_scores["score"]["std"], name="Standard Deviation"),
        row=1, col=2
    )

    fig.add_trace(
        go.Bar( x=post_scores.index, y=post_scores["score"]['max'], text=post_scores["score"]["max"], name="Max"),
        row=2, col=1
    )


    fig.add_trace(
            go.Bar( x=post_scores.index, y=post_scores["score"]['min'], text=post_scores["score"]["min"], name="Min"),
        row=2, col=2
    )


    fig.update_layout(height=1000, width=1000)
    st.plotly_chart(fig)


    fig = make_subplots(rows=2, cols=2)

    fig.add_trace(
        go.Bar( x=comment_scores.index, y=comment_scores["score"]['mean'], text=comment_scores["score"]["mean"], name="Mean"),
        row=1, col=1
    )

    toc.h3("Comments")
    fig.add_trace(
        go.Bar( x=comment_scores.index, y=comment_scores["score"]['std'], text=comment_scores["score"]["std"], name="Standard Deviation"),
        row=1, col=2
    )

    fig.add_trace(
        go.Bar( x=comment_scores.index, y=comment_scores["score"]['max'], text=comment_scores["score"]["max"], name="Max"),
        row=2, col=1
    )


    fig.add_trace(
            go.Bar( x=comment_scores.index, y=comment_scores["score"]['min'], text=comment_scores["score"]["min"], name="Min"),
        row=2, col=2
    )


    fig.update_layout(height=1000, width=1000,)
    st.plotly_chart(fig)


    toc.h3("Posts by subreddit")
    st.write("This graph shows the amount of posts collected from each of the subreddits selected. The target was to collect the top 1000 posts. For some of them there are fewer.")
    plot_category_counts(posts)

    toc.h3("Comments by subreddit")
    st.write("Here we also have the distribution of comments collected from each of the subreddits. LGBT sub is one of the most populated and this is why there are more discussions.")

    plot_category_counts(comments)

    ## SIDEBAR WIDGETS
    selected_category  = st.sidebar.selectbox("Select community", options = categories)
    k = int( np.floor(st.sidebar.number_input('Insert number of top n-grams to show', value = 50, step=1, min_value=1)))
    n = int( np.floor(st.sidebar.number_input('Insert what type of n-grams to analyze', value = 3, step=1, min_value=1, max_value=6)))


    toc.h2("Subreddits's sentiments comparison")
    st.write("Sentiment analysis is a process of computationally determining the emotional tone behind a piece of text. It involves analyzing the language used in a sentence, paragraph, or document to understand whether the expressed sentiment is positive, negative, or neutral. This analysis is commonly used in social media monitoring, customer feedback analysis, and market research to gauge public opinion, understand customer satisfaction, or track brand perception.")
    st.write("""Here a sentiment analysis model was applied to the collected posts and comments.
             This model assigns probabilities to each given text of being:\n- Positive, negative or neutral\n- Hate speech detection (hatefulness, aggressiveness)\n- Emotions: joy, sadness, surprise, disgust, fear, anger, other \n
             """)
    st.write("In the folowing graphs we can see the average probarbilities given to each subreddit:")
    toc.h3("Posts")
    plot_sentiments(posts)
    toc.h3("Comments")
    plot_sentiments(comments)

    ## PUBLICATION HISTORY
    toc.h2("Publication History")
    st.write(f"This section shows the frequency of posts and comments publication of the subreddit **{selected_category}**")
    toc.h3("Posts {} ".format(selected_category))
    plot_history(posts_date_agg, selected_category)
    toc.h3("Comments {} ".format(selected_category))
    plot_history(comments_date_agg, selected_category)


    title = "Top Comments: {} ".format(selected_category) 
    st.write(f"The following table contain the top comments of **{selected_category}**. This can give us a way to see which types of messages are posted in the sub and which of them are more valued. Note: Hover over the table to perform a keyword search.")
    toc.h2(title)
    print_top_comments(selected_category)

    toc.h2("Top N-grams")
    st.write("Topic analysis, also known as topic modeling, is a technique used in natural language processing to identify the main themes or topics present in a collection of documents. It involves algorithms that analyze the words and phrases within the documents to automatically uncover common themes or topics. These topics are represented as sets of words that frequently co-occur together within the documents.")
    st.write("N-grams are a way to implement topic analysis. An n-gram is a sequence of n words and one way of identifying popular topics is listing the ones are more repeated.")
    st.write(f"These are the frequencies for the **top {k} {n}-grams for {selected_category}**:")
    toc.h3("Posts {} ".format(selected_category))
    top_k_ngrams(posts[posts['category']==selected_category], K=k, N_1=n, N_2=n)

    toc.h3("Comments {}".format(selected_category))
    top_k_ngrams(comments[comments['category']==selected_category], K=k, N_1=n, N_2=n)

    toc.h2("Wordclouds")
    st.write("Wordclouds are a very popular representation to show the most repeated n-grams. All the sequences are displayed together and the larger they appear, the more frequent they are.")
    st.write(f"These are the wordclouds for the **top {k} {n}-grams for {selected_category}**:")
    toc.h3("Posts {} ".format(selected_category))
    ngram_wordcloud(posts[posts['category']==selected_category], K=k, N_1=n, N_2=n)

    toc.h3("Comments {}".format(selected_category))
    ngram_wordcloud(comments[comments['category']==selected_category], K=k, N_1=n, N_2=n)
    toc.toc()

    # subreddits = [selected_category]
    # keywords=['']
    # display_table(subreddits = subreddits, keywords = keywords)


    # selected_comments = comments[comments['category']==selected_category]['clean_text']
    # selected_posts = posts[posts['category']==selected_category]['clean_text']
    # st.plotly_chart(LDA(pd.concat([selected_posts, selected_comments])))   


