#!/usr/bin/env python
# coding: utf-8

import redis
import pickle
from gensim.summarization import keywords
from urllib.request import urlopen as url
from progress.bar import Bar
DEBUG = False
r = redis.StrictRedis(host = 'localhost', port = 6379, db = 0)

def publish(keyword, ID_news):
    """
    inputs: keyword = string
            ID_news = id of the news article
    """
    #print("alert all users -> new article on this keyword")
    pass
def set_news(body, keywords):
    """
    inputs: body = (long) string
            keywords = list of string
    """
    keywords = list(set(keywords))
    ID_news = int(r.incr('ID_article'))
    r.hset("news:"+str(ID_news),"body", body)
    for key in keywords:
        if(int(r.sismember("keywords", key)) == 0):
            r.sadd("keywords", key)
        r.sadd("keywords:" + key, ID_news) # add every articles in each keyword set
        publish(key, ID_news)
    return ID_news

def get_news(ID_user):
    """
    input : ID_user
    output : list of news_id corresponding to its subcriptions.
    """
    keywords_user = r.smembers("user_keyword:" + str(ID_user))
    ID_news_adapted = []
    news_adapted = []
    for key in keywords_user:
        if(DEBUG):
            print("keywords:"+key.decode("utf-8"))
            print( r.sismember("keywords",str(key)))
            print( r.smembers("keywords:"+str(key)))
        for members in r.smembers("keywords:"+ key.decode("utf-8")):
            if(DEBUG):
                print(members)
            ID_news_adapted.append(int(members.decode("utf-8")))
    sorted = list(set(ID_news_adapted))
    sorted.sort()
    return sorted

def get_IDuser():
    return  int(r.incr('ID_user'))

def subscribe(ID_user,keyword):
    r.sadd("user_keyword:" + str(ID_user), keyword)

def unsubscribe(ID_user,keyword):
    r.srem("user_keyword:" + str(ID_user), keyword)

def get_subscriptions(ID_user):
    return r.smembers("user_keyword:" + str(ID_user))

def get_subjects():
    return r.smembers("keywords")

def add_news(wiki_url):
    """
    input : an url from wikipedia
    action : add the url to the bdd and its computed keywords
    """
    with url(wiki_url) as response:
        html = response.read()
        wiki_keywords = keywords(html).split('\n')
        wiki_keywords = wiki_keywords[:10]
        ID_news = set_news(html,wiki_keywords)
    return ID_news, wiki_keywords


def seed():
    """
    action : add multiple urls to the bdd and its computed keywords
            and show a progress bar
    """
    urls = ["https://fr.wikipedia.org/wiki/Redis",
        "https://fr.wikipedia.org/wiki/C_(langage)",
        "https://fr.wikipedia.org/wiki/Licence_BSD",
        "https://fr.wikipedia.org/wiki/NoSQL"
       ]
    bar = Bar('seed of redis database', max=len(urls))
    for wiki_url in urls:
        with url(wiki_url) as response:
            html = response.read()
            wiki_keywords = keywords(html).split('\n')
            set_news(html,wiki_keywords[:10])
            bar.next()
    bar.finish()



def quick_test():
    """
    tests used to developp small functions
    """
    set_news("test",["1","2","3"])
    get_subjects()
    r.smembers("keywords")
    subscribe("1")
    get_subscriptions(2)
    unsubscribe(2,1)
    get_subscriptions(2)
