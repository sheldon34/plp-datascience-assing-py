import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import re

sns.set()

st.set_page_config(layout='wide')
st.title('CORD-19 Data Explorer')
st.write('Simple exploration of CORD-19 metadata (metadata.csv)')

@st.cache_data
def load_data(path='metadata_clean.csv'):
    return pd.read_csv(path)

# Load data (expects metadata_clean.csv in the same folder)
df = None
try:
    df = load_data()
except Exception as e:
    st.error(f"Could not load cleaned metadata CSV: {e}. Please run the notebook to produce 'metadata_clean.csv' or place it next to this app.")

if df is not None:
    # Sidebar filters
    st.sidebar.header('Filters')
    years = sorted([int(y) for y in df['year'].dropna().unique()]) if 'year' in df.columns else []
    if years:
        yr_min, yr_max = st.sidebar.select_slider('Year range', options=years, value=(min(years), max(years)))
        df = df[(df['year'] >= yr_min) & (df['year'] <= yr_max)]

    journal_col = None
    for candidate in ['journal', 'journal_title', 'journal_ref', 'source_x', 'source_y']:
        if candidate in df.columns:
            journal_col = candidate
            break

    top_n = st.sidebar.slider('Top N journals', 5, 50, 15)

    st.header('Summary')
    st.write('Rows in view:', len(df))

    col1, col2 = st.columns([2,3])

    with col1:
        st.subheader('Publications by Year')
        if 'year' in df.columns:
            year_counts = df['year'].value_counts().sort_index()
            fig, ax = plt.subplots(figsize=(6,3))
            year_counts.plot(kind='bar', ax=ax)
            ax.set_ylabel('Count')
            st.pyplot(fig)
        else:
            st.info('No year column available in data.')

    with col2:
        st.subheader('Top Journals')
        if journal_col:
            top_journals = df[journal_col].fillna('Unknown').value_counts().head(top_n)
            fig2, ax2 = plt.subplots(figsize=(6,4))
            sns.barplot(x=top_journals.values, y=top_journals.index, ax=ax2)
            st.pyplot(fig2)
        else:
            st.info('No journal-like column found.')

    st.subheader('Word Cloud of Titles')
    if 'title' in df.columns:
        text = ' '.join(df['title'].dropna().astype(str).tolist()).lower()
        # basic cleaning
        text = re.sub(r"[^a-z0-9\\s]", ' ', text)
        wc = WordCloud(width=800, height=300, background_color='white').generate(text)
        fig3, ax3 = plt.subplots(figsize=(10,3))
        ax3.imshow(wc, interpolation='bilinear')
        ax3.axis('off')
        st.pyplot(fig3)
    else:
        st.info('No title column to generate word cloud.')

    st.subheader('Sample rows')
    st.dataframe(df.sample(min(100, len(df))))

    st.write('---')
    st.write('Notes: this app expects `metadata_clean.csv` produced by the notebook to be in the same directory as the app.')
