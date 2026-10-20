# 🔎 AI Research Agent

## Overview
AI Research Agent is a Streamlit web app that answers questions by searching the web in real time. It uses a LangChain agent powered by Groq's `gpt-oss-120b` model and DuckDuckGo search, then returns a concise answer with sources.

**Live demo:** _add your Streamlit URL here_

## Features
- Ask any question and get a researched answer
- Live web search through DuckDuckGo
- Example questions and answer history
- Simple, beginner-friendly interface

## Tech Stack
- [Streamlit](https://streamlit.io) – frontend
- [LangChain](https://www.langchain.com) – agent framework
- [Groq](https://groq.com) – LLM (`openai/gpt-oss-120b`)
- DuckDuckGo – web search

## Run Locally

```bash
git clone https://github.com/YOUR_USERNAME/ai-research-agent.git
cd ai-research-agent
pip install -r requirements.txt
```

Create a `.env` file:

```
GROQ_API_KEY=your_key_here
```

Start the app:

```bash
streamlit run app.py
```

## Deploy
1. Push the repo to GitHub.
2. Create an app on [Streamlit Community Cloud](https://share.streamlit.io).
3. Add `GROQ_API_KEY` under **Settings → Secrets**.

## Project Structure

```
ai-research-agent/
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
└── notebooks/agent-langchain-research.ipynb
```
