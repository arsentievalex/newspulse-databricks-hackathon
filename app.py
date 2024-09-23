import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from streamlit.components.v1 import html
import time
# from databricks import sql
import numpy as np
from collections import defaultdict
from streamlit_lightweight_charts import renderLightweightCharts
from yahooquery import Ticker
import datetime
from embedchain import App
from embedchain.config import BaseLlmConfig
import os
import functions
from yahooquery import search
import json
import requests


# wide streamlit format
st.set_page_config(layout='wide')

# read text from index.txt
with open('index.html', 'r') as file:
    html_content = file.read()

html(html_content, height=250)

# create a session state for login
# if 'logged_in' not in st.session_state:
#     st.session_state['logged_in'] = False

# watchlist = []

server_hostname = "YOUR_SERVER_HOSTNAME"
http_path = "YOUR HTTP_PATH"
access_token = "YOUR_ACCESS TOKEN"


# establish connection to databricks db
# connection = functions.get_conn(server_hostname, http_path, access_token)

with st.sidebar:
    with st.form(key='login_form'):
        username = st.text_input(label='Username')
        password = st.text_input(label='Password', type='password', placeholder='********')
        submit_button = st.form_submit_button(label='Log In')
    st.info("Logging in has been disabled for demo purposes")

# if submit_button and username != "" and password != "":

#     with st.sidebar:
#         with st.spinner("logging you in..."):
#             user_row = functions.find_user(connection, username, password)

#     if len(user_row) == 0:
#         st.error('Invalid username or password')
#         st.stop()
#     else:
#         get company that a user added to their watchlist
#         watchlist = user_row[0].watchlist

#         with st.sidebar:
#             st.write("You're logged in as: ", username)

#         st.session_state['logged_in'] = True

# FOR DEV
st.session_state['logged_in'] = True
watchlist = "Tesla"

if st.session_state['logged_in']:

    # load articles associated with the company
    # articles = functions.get_data(connection, watchlist)
    
    # convert to df
    # articles_df = pd.DataFrame(articles)
    articles_df = pd.read_csv('articles.csv')

    # close connection
    # connection.close()

    # rename columns to be: url, content, company_name, date, sentiment
    articles_df.columns = ['url', 'content', 'company_name', 'date', 'sentiment']

    # convert watchlist to list
    watchlist = watchlist.split(',')
    st.multiselect("Your Watchlist", watchlist, default=watchlist)

    tab1, tab2, tab3 = st.tabs(['Sentiment Analysis', 'Stock Price vs Sentiment', 'Chatbot'])

    # sentiment analysis tab
    with tab1:
        st.info('Below heatmaps present the sentiment analysis of the most recent news articles. The range of sentiment is from -1 to 1, where -1 is negative sentiment, 0 is neutral sentiment, and 1 is positive sentiment.')
        
        # replace null with None
        articles_df['sentiment'] = articles_df['sentiment'].apply(lambda x: x.replace('null', 'None'))

        # put sentiment column into a list
        sentiment_data = articles_df['sentiment'].tolist()

        # force convert to dict
        clean_sentiment_list = [eval(x) for x in sentiment_data]

        agg_df = functions.aggregate_sentiment(clean_sentiment_list)

        # keep only the date and Sentiment columns
        date_df = functions.transform_sentiment(articles_df[['date', 'sentiment']])

        # columns to list
        columns = date_df.columns.tolist()

        # drop sentiment topic from columns
        columns.remove('Sentiment Topic')

        # Apply gradient coloring
        styled_date_df = date_df.style.background_gradient(
            cmap="RdYlGn",
            subset=columns,
            vmin=-1,
            vmax=1
        ).format("{:.2f}", subset=columns)

        styled_agg_df = agg_df.style.background_gradient(
            cmap="RdYlGn",
            subset=['Sentiment Score'],
            vmin=-1,
            vmax=1
        ).format("{:.2f}", subset=['Sentiment Score'])

        col1, col2 = st.columns(2)

        with col1:
            st.dataframe(styled_date_df, hide_index=True, use_container_width=True)
        with col2:
            st.dataframe(styled_agg_df, hide_index=True, use_container_width=True)


    # stock price vs sentiment tab
    with tab2:

        st.info("The histogram shows the sentiment score of the articles published on a given date. The color represents negative or positive sentiment and the value is intensity (0-100)."
                "The stock price is plotted on the area chart.")
    
        # load stock price data
        # tkr = functions.get_ticker(watchlist[0])
        
        price_series = functions.get_stock_history('TSLA', '30d', '1d')

        priceVolumeSeriesHistogram = functions.transform_date_sentiment(date_df)

        functions.plot_chart(price_series, priceVolumeSeriesHistogram)

    # chatbot tab. For demo purposes will use embedchain and a few sample news articles.
    with tab3:
        st.info("The chatbot is trained only on selected articles for demo purposes")

        urls = ["https://www.msn.com/en-us/autos/news/tesla-s-supercharger-layoffs-couldn-t-have-come-at-a-worse-time/ar-AA1o6uYb",
                "https://www.msn.com/en-us/money/news/i-landed-a-dream-internship-at-tesla-now-im-scrambling-after-the-company-cancelled-my-internship-3-weeks-before-i-was-set-to-start/ar-AA1o3OFp",
                "https://www.wired.com/story/tesla-supercharger-pullback-filling-the-power-gap/",
                "https://www.ft.com/content/114effb2-1071-4d93-b53d-00a96a0336a2",
                "https://www.msn.com/en-us/money/companies/elimination-of-teslas-charging-department-raises-worries-as-evs-from-other-automakers-join-network/ar-AA1nZzGg",
                "https://www.msn.com/en-us/money/companies/tesla-lays-off-hundreds-of-employees-on-electric-vehicle-charger-team/ar-AA1nZsPe",
                "https://www.msn.com/en-us/autos/news/tesla-staff-say-entire-supercharger-team-fired/ar-AA1nYAl8",
                "https://www.msn.com/en-us/money/other/tesla-retreat-from-ev-charging-leaves-growth-of-u-s-network-in-doubt/ar-AA1o64CD",
                "https://arstechnica.com/cars/2024/05/chaos-at-tesla-what-analysts-think-about-elon-musks-cuts-and-layoffs/"
                ]

        # set openai api key
        os.environ["OPENAI_API_KEY"] = st.secrets["openai_credentials"]["API_KEY"]

        bot = functions.load_bot(urls)
        query_config = BaseLlmConfig(number_documents=1)

        if "messages" not in st.session_state.keys():  # Initialize the chat messages history
            st.session_state.messages = [
                {"role": "assistant", "content": "Ask me a question!"}]

        if prompt := st.chat_input("Your question"):  # Prompt for user input and save to chat history
            st.session_state.messages.append({"role": "user", "content": prompt})

        for message in st.session_state.messages:  # Display the prior chat messages
            # if role is user
            if message["role"] == "user":
                with st.chat_message(message["role"]):
                    st.write(message["content"])
            elif message["role"] == "assistant":
                with st.chat_message(message["role"]):
                    st.write(message["content"])

        # If last message is not from assistant, generate a new response
        if st.session_state.messages[-1]["role"] != "assistant":
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    response, citations = bot.chat(prompt, citations=True, config=query_config)

                    sources = functions.get_sources(citations)
                    # italicized_sources = [f"*{source}*" for source in sources]

                    full_response = response + "\n\n**Source**:\n" + f"*{sources[0]}*"

                    st.write(full_response)

                    message = {"role": "assistant", "content": full_response}
    #                 st.session_state.messages.append(message)  # Add response to message history


