# NewsPulse AI: Databricks Generative AI Hackathon

The app is designed to monitor the latest news articles and analyze sentiment around various business events, such as layoffs, M&As, reorgs, disputes, etc. Such events may have significant impact on stock performance, and therefore are crucial for the investors.

The app has three main features: Sentiment Analysis (by day and topic & aggregated) Stock Price vs Sentiment (time series that allows to analyze impact of news sentiment on stock performance) Chatbot (Q&A with vector search index and sources)

The process of acquiring the data is as follows:

* DuckDuckGo API is used to fetch the recent news articles about the selected company.
* ScrapeGraphAI and GPT 3.5-Turbo is used to scrape article content from URLs.
* DBRX Instruct and LangChain is used to extract sentiment from articles.
* RAG: the articles are split into chunks, embedded & loaded to vector store.
* YahooQuery is used to load stock price history data.

The idea is that the Databricks jobs are scheduled to run every day or even multiple times per day to enrich the database and vector store with the newest articles.

<img src="https://i.postimg.cc/hvqBYt93/newspulse.gif"/>
