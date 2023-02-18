# ---
# lambda-test: false
# ---


from fastapi.encoders import jsonable_encoder
from typing import Optional

from fastapi import FastAPI, Header
from pydantic import BaseModel

import modal

web_app = FastAPI()

image = (
    modal.Image.debian_slim()
    .run_commands(
        "apt-get update",
    )
    .pip_install("bs4", "youdotcom", "requests")
)

stub = modal.Stub("modal-serverless-api", image=image)


class URL(BaseModel):
    url_link: str


@stub.function
def extract_source(url):
    start = url.find("www.") + 4
    end = url.find(".com")
    source = url[start:end]
    return source


@stub.function
def search_youdotcom(url):
    import requests
    import os
    import json
    from bs4 import BeautifulSoup
    import time
    from youdotcom import Search

    article_response = requests.get(url, timeout=10)
    soup = BeautifulSoup(article_response.text, "html.parser")
    article = soup.get_text(" ", strip=True)

    title = soup.find("title").text
    search_results = Search.search_for(title)

    parsed = json.loads(search_results["results"])
    print(json.dumps(parsed, indent=4))


@stub.function(secret=modal.Secret.from_name("modal_serverless_api_secrets"))
def search_initial_article(url):

    import requests
    import os
    import json
    from bs4 import BeautifulSoup
    import time

    API_KEY = os.environ["GOOGLE_API_KEY"]
    ENGINE_ID = os.environ["ENGINE_ID"]

    # Get the article text and title from the URL
    article_response = requests.get(url)
    soup = BeautifulSoup(article_response.text, "html.parser")
    article = soup.get_text(" ", strip=True)

    title = soup.find("title").text

    # Search the web for similar articles using the article title
    response = requests.get(
        "https://www.googleapis.com/customsearch/v1?q="
        + title
        + "&cx="
        + ENGINE_ID
        + "&key="
        + API_KEY
    )
    print(response.json())
    results = response.json()["items"]

    comparison_articles = {}
    comparison_articles[url] = article

    sources = []
    sources.append(extract_source.call(url))

    for result in results[:5]:
        if extract_source.call(result["link"]) not in sources:
            sources.append(extract_source.call(result["link"]))
            article_response = requests.get(result["link"])
            soup = BeautifulSoup(article_response.text, "html.parser")
            article = soup.get_text(" ", strip=True)
            comparison_articles[result["link"]] = article

            print("Comparison article:", result["link"])

    print(json.dumps(comparison_articles, indent=4))

    return jsonable_encoder(comparison_articles)


@web_app.post("/search/")
async def search(url: URL):
    return search_initial_article.call(url.url_link)


@stub.asgi(image=image)
def fastapi_app():
    return web_app


if __name__ == "__main__":
    stub.deploy("webapp")
