from langchain_community.document_loaders import WebBaseLoader
import os

def load_websites(urls):
    loader = WebBaseLoader(
        urls,
        requests_kwargs={"headers": {"User-Agent": os.getenv("USER_AGENT", "MyApp/1.0")}}
    )
    return loader.load()