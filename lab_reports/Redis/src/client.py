#!/usr/bin/env python
# coding: utf-8

import socket
import time
import json
from termcolor import colored
import pickle

HOST, PORT, BUF = '', 8844, 1024

# get help command / check if server is active

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.connect((HOST, PORT))
data_init = {"ID_user":'0','message':'get_IDuser'}
s.send(pickle.dumps(data_init, -1) )
time.sleep(1)
ID_user = s.recv(BUF).decode('utf8')

print("your ID user is :",colored(ID_user, 'red'), "please keep it to manage your account")
data = {"ID_user":ID_user,'message':'get_IDuser'}
print("Press", colored("help",'blue'),"to get all commands")

while 1:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect((HOST, PORT))
    input_cmd = input()
    data['message'] = input_cmd
    s.send(pickle.dumps(data, -1) )
    print(colored("Sent","green"))
    time.sleep(1)
    print(colored("Server: ","red") + s.recv(BUF).decode('utf8'))
