# ---
# lambda-test: false
# ---


from fastapi.encoders import jsonable_encoder
from typing import Optional

from fastapi import FastAPI, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import modal

web_app = FastAPI()


origins = ["*"]

web_app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

image = (
    modal.Image.debian_slim()
    .apt_install("curl")
    .run_commands(
        "apt-get update",
        "curl -O https://raw.githubusercontent.com/TruthQuestWeb/ml-model/main/train.csv",
    )
    .pip_install(
        "bs4", "youdotcom", "requests", "openai", "pandas", "scikit-learn", "numpy"
    )
)

# Define the filename and path to save the pickled classifier
filename = "classifier.pickle"
volume = modal.SharedVolume().persist("model-cache-vol")

stub = modal.Stub("modal-serverless-api", image=image)
# 1 day duration
CACHE_DURATION = 86400
CACHE_DIR = "/cache"


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
def summarizer(url):
    import os
    import openai
    import requests
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

    openai.api_key = os.getenv("OPENAI_API_KEY")

    response = openai.Completion.create(
        model="text-davinci-003",
        prompt=article + "\n\n Remove opinion and summarize the article.",
        temperature=0.36,
        max_tokens=200,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=1,
    )

    print(response)

    return jsonable_encoder(response)


@stub.function(memory=4048)
def foo(articles):
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.model_selection import train_test_split
    from sklearn.naive_bayes import MultinomialNB
    from sklearn import metrics

    import pickle
    import pandas as pd
    import time

    df = pd.read_csv("/train.csv")

    import numpy as np

    # Check if the pickled file exists and if it has not expired
    try:
        with open(filename, "rb") as f:
            classifier, cache_time = pickle.load(f)
            if time.time() - cache_time < CACHE_DURATION:
                df = pd.DataFrame({"text": [articles]})
                input = count_vectorizer.transform(df)

                # Make predictions and return the results
                pred = nb_classifier.predict(input)
                if pred == 1:
                    return jsonable_encoder({"result": "true"})
                else:
                    return jsonable_encoder({"result": "false"})

    except FileNotFoundError:
        pass

    df["text"].replace("", np.nan, inplace=True)
    df.dropna(subset=["text"], inplace=True)

    X_train, X_test, y_train, y_test = train_test_split(
        df.text, df.label, test_size=0.2
    )

    count_vectorizer = CountVectorizer(stop_words="english")
    count_train = count_vectorizer.fit_transform(X_train)
    count_test = count_vectorizer.transform(X_test)

    nb_classifier = MultinomialNB()
    nb_classifier.fit(count_train, y_train)

    with open(filename, "wb") as f:
        pickle.dump((nb_classifier, time.time()), f)

    df = pd.DataFrame({"text": [articles]})
    input = count_vectorizer.transform(df)

    pred = nb_classifier.predict(input)
    if pred == 1:
        return jsonable_encoder({"result": "true"})
    else:
        return jsonable_encoder({"result": "false"})


class Article(BaseModel):
    text: str

@stub.function(memory=4048, cpu=4.0)
def confidence(articles):
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.model_selection import train_test_split
    from sklearn.naive_bayes import MultinomialNB

    import pandas as pd
    df = pd.read_csv('/train.csv')

    import numpy as np
    df[['title', 'author']].replace('', np.nan, inplace=True)

    df.dropna(subset=['title'], inplace=True)
    df.dropna(subset=['author'], inplace=True)

    df['content'] = df['author'] + ' ' + df['title']

    X_train, X_test, y_train, y_test = train_test_split(df.content, df.label, test_size=.2)

    count_vectorizer = CountVectorizer(stop_words='english')
    count_train = count_vectorizer.fit_transform(X_train)

    nb_classifier = MultinomialNB()
    nb_classifier.fit(count_train, y_train)

    df = pd.DataFrame([articles ])
    input = count_vectorizer.transform(df)
    pred = nb_classifier.predict_proba(input)
    return jsonable_encoder({'true': pred[0][1], 'false': pred[0][0]})

@web_app.post("/analysis/")
async def analysis(articleobj: Article):
    return confidence.call(articleobj.text)


@stub.function(
    secret=modal.Secret.from_name("modal_serverless_api_secrets"), memory=12288, cpu=6.0
)
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

    # Get the author of the article
    author = soup.find("meta", property="article:author")

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
    comparison_articles["title"] = title
    comparison_articles["author"] = author

    #result = foo.call(comparison_articles["text"])

    confi = confidence.call(comparison_articles)
    #merged_data = {**result, **confi}

    return jsonable_encoder(confi)


"""
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
"""


@web_app.post("/search/")
async def search(url: URL):
    art = search_initial_article.call(url.url_link)

    return jsonable_encoder(art)


@web_app.post("/summarize/")
async def summarize(url: URL):
    return summarizer.call(url.url_link)


# @web_app.post("/video/")
# async def video(url: URL):
#   return search_youdotcom.call(url.url_link)


@stub.asgi(image=image)
def fastapi_app():
    return web_app


if __name__ == "__main__":
    stub.deploy("webapp")
