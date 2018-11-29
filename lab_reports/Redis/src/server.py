#!/usr/bin/env python
# coding: utf-8
# code adapted from http://apprendre-python.com/page-reseaux-sockets-python-port
import socket
import sys
import threading
from news import subscribe, unsubscribe, get_subscriptions, get_subjects, get_IDuser
from news import publish, set_news, add_news, get_news
from news import seed
import pickle
import re

HOST, PORT = '', 8844

class ClientThread(threading.Thread):
    def __init__(self, ip, port, clientsocket):
        threading.Thread.__init__(self)
        self.ip = ip
        self.port = port
        self.clientsocket = clientsocket
        print("[+] new thread for %s %s" % (self.ip, self.port, ))


    def run(self):
        print("Connection of %s %s" % (self.ip, self.port, ))
        r = self.clientsocket.recv(2048)
        data = pickle.loads(r)
        ID_user, message = data['ID_user'], data['message']
        print(ID_user,message)
        subscribe_str = r"subscribe\(([\w\"]*)\)"
        unsubscribe_str = r"unsubscribe\(([\w\"]*)\)"
        add_news_str = r"add_news\(([]\"]?http[s]?://[\w\"\.\/]*)\)"

        if(message == "help" or
            message == "help()") :
            self.clientsocket.send(b' '+ b'get_IDuser will give you an userId' + b' ')
            self.clientsocket.send(b'\n ')
            self.clientsocket.send(b' '+ b'get_subjects will give you a list of keyword' + b' ')
            self.clientsocket.send(b'\n ')
            self.clientsocket.send(b' '+ b'get_subscriptions() will return the list of your subscriptions' + b' ')
            self.clientsocket.send(b'\n ')
            self.clientsocket.send(b' '+ b'unsubscribe(keyword) will remove the keyword subscription' + b' ')
            self.clientsocket.send(b'\n ')
            self.clientsocket.send(b' '+ b'subscribe(keyword) will remove the keyword subscription' + b' ')
            self.clientsocket.send(b'\n ')
            self.clientsocket.send(b' '+ b'add_news(url) will add the news and its keyword from the url' + b' ')
            self.clientsocket.send(b'\n ')
            self.clientsocket.send(b' '+ b'get_news will give all the news for you' + b' ')
            self.clientsocket.send(b'\n ')
            self.clientsocket.send(b' '+ b'note: only tested on wikipedia url' + b' ')

        elif(message=="get_subjects" or
           message== "get_subjects()" ):
            subjects = get_subjects()
            for sub in subjects:
                self.clientsocket.send(b' '+ sub + b' ')

        elif(message=="get_IDuser" or
           message== "get_IDuser()" ):
            current_id = get_IDuser()
            self.clientsocket.send(str(current_id).encode('utf-8'))

        elif(message=="get_subscriptions" or
            message== "get_subscriptions()" ):

            subscriptions = get_subscriptions(ID_user)
            self.clientsocket.send(b' your subcriptions: \n' + b' ')
            for sub in subscriptions:
                self.clientsocket.send(b' '+ sub + b' \n ')

        elif(message=="get_news" or
            message== "get_news()" ):
            ID_news_adapted = get_news(ID_user)
            self.clientsocket.send(b' your adapted news: \n' + b' ')
            for id in ID_news_adapted:
                self.clientsocket.send(b' news with id:'+ str(id).encode('utf-8') + b' \n ')

        elif(re.search(subscribe_str,message)!= None):

            keyword = re.search(subscribe_str,message).groups()[0]
            print(ID_user,keyword.encode('utf-8'))
            subscriptions = subscribe(int(ID_user), keyword.encode('utf-8'))
            self.clientsocket.send(b'subcribed to : \n' +keyword.encode('utf-8') + b' successfully \n')

        elif(re.search(unsubscribe_str,message)!= None):
            keyword = re.search(unsubscribe_str,message).groups()[0]
            print(int(ID_user).encode('utf-8'),keyword.encode('utf-8'))
            subscriptions = unsubscribe(int(ID_user),keyword.encode('utf-8'))
            self.clientsocket.send(b'unsubcribed to : \n' +keyword.encode('utf-8') + b' successfully \n')

        elif(re.search(add_news_str,message)!= None):
            url = re.search(add_news_str,message).groups()[0]
            ID_news,keyword = add_news(url.strip('"'))
            publish(ID_news,keyword)
            self.clientsocket.send(b'added news from : \n' +url.encode('utf-8') + b' successfully \n')
        else:
            self.clientsocket.send(b' '+b'Enter help to see the commands'+ b'\n ')

tcpsock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
tcpsock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
tcpsock.bind((HOST, PORT))
seed()
while True:
    tcpsock.listen(10) # up to 10 clients
    print( "Listening ...")
    (clientsocket, (ip, port)) = tcpsock.accept()
    newthread = ClientThread(ip, port, clientsocket)
    newthread.start()
