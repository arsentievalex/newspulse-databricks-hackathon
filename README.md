[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://newspulseai.streamlit.app/)

# NewsPulse AI: Databricks Generative AI Hackathon [1st place winner in Financial Services]

## What It Does
This application is specifically designed to monitor and analyze the sentiment of the latest news articles regarding significant business events, such as layoffs, mergers and acquisitions, reorganizations, and disputes. These events can profoundly affect stock performance, making it vital for investors to stay informed.

### Key Features
- **Sentiment Analysis:** Analyze sentiment by day and topic, with aggregated results.
- **Stock Price vs Sentiment:** A time series analysis to study the impact of news sentiment on stock performance.
- **Chatbot:** Provides Q&A capabilities using a vector search index and sourced information.

### Data Acquisition Process
- **News Articles:** Uses the DuckDuckGo API to fetch recent news articles about selected companies.
- **Content Scraping:** Utilizes ScrapeGraphAI and GPT 3.5-Turbo to extract content from URLs.
- **Sentiment Extraction:** Applies DBRX Instruct and LangChain to determine sentiment from articles.
- **RAG System:** Articles are chunked, embedded using DBRX, and loaded into a Databricks vector store.
- **Stock Data:** Uses YahooQuery to gather historical stock price data from YahooFinance.

Automated Databricks jobs are supposed to run daily or multiple times a day to continuously update the database and vector store with new articles.

<img src="https://i.postimg.cc/hvqBYt93/newspulse.gif"/>

## Tech Stack
- [Databricks](https://www.databricks.com/) - Data Processing, Storage, Vector Database
- [Streamlit](https://streamlit.io/) - Frontend
- [OpenAI](https://www.openai.com/) - LLM
- [DBRX](https://www.databricks.com/blog/introducing-dbrx-new-state-art-open-llm) - LLM
- [Langchain](https://js.langchain.com/docs/) - LLM wrapper
- [DuckDuckGo](https://rapidapi.com/epctex-epctex-default/api/duckduckgo10/) - News API
- [ScrapeGraphAI](https://github.com/VinciGit00/Scrapegraph-ai/tree/main) - Web Scraping
- [Yahooquery](https://yahooquery.dpguthrie.com/) - Yahoo Finance API
- [Embedchain](https://embedchain.ai/) - RAG (used for demo as alternative to Databricks endpoint)


<img src="https://i.postimg.cc/BnKW2WsN/newspulse-architecture.png"/>
