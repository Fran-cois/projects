import os
import psutil
import time
import numpy as np
import matplotlib as mpl

if os.environ.get('DISPLAY','' ) == '':
    print('no display found. Using non-interactive Agg backend')
    mpl.use('Agg')
import matplotlib.pyplot as plt
import psutil
import random
import string
import unittest
import sys


def get_failure_transition(path):
    path = np.array(path)
    fail_array = [0]
    for i in range(1,len(path)+1):
        j = 0
        while(j < i and i >= 1 and np.array_equal(path[:j] , path[i-j:i])):
            j+=1
        if(j > 0):
            j -= 1
        fail_array.append(j)
    return fail_array


# In[36]:

def lazy_dfa(file,path):
    # get the pi
    cuts = [0]
    for i in range(len(path)):
        if (path[i] == '' or path[i] == ' ' ):
            cuts.append(i)
    cuts.append(len(path))
    #DEBUG_print("cuts",cuts)
    # get the number of state
    number_of_state = len(path) - len(cuts) - 2
    # get failure transition array modified.
    fail_array = []

    for i in range(len(cuts) - 1 ):
        indice_i  = cuts[i]
        indice_j = cuts[i+1]
        fail_tmp = get_failure_transition(path[indice_i:indice_j])
        fail_tmp  = [x + indice_i for x in fail_tmp]
        fail_array += fail_tmp

    #DEBUG_print("fail_array", fail_array)

    # replace path[i]  == machin by custom_path[i]

    #DEBUG_print("path", path)
    new_path = list(filter(('').__ne__, path))

    #DEBUG_print("newpath",new_path)
    path = new_path
    S = []
    state_i = [0]
    i = 0
    count = 0
    Results = []
    with open(file,'r') as f:
        for line in f :
            begin, current_el = line.split(' ')[0:2]
            current_el = current_el.replace('\n','')
            i = state_i[-1]
            #DEBUG_print(S,i,count, state_i)
            if(i == len(path)):
                if(S[-1] not in Results):
                    Results.append(S[-1])
                i = fail_array[i]

            if(begin == '0'): # true, start Element
                S.append(count)
                count +=  1
                if(path[i] == current_el):
                    i += 1
                else:                    #do failure transition
                    i = fail_array[i]
                    if(path[i] == current_el):
                        i+=1
                state_i.append(i)

            else:
                if(S):
                    S.pop()
                    state_i.pop()

    return Results
if __name__ == '__main__':
    file = sys.argv[1]
    query = sys.argv[2]
    path = query[2:].split('/')
    print(lazy_dfa(file,path))
