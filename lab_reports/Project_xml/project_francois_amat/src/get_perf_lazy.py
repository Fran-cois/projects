
from  lazy_dfa import *


def get_memory_usage():
    pid = os.getpid()
    py = psutil.Process(pid)
    #memoryUse = py.memory_info()[0]/2.**36  # memory use in kB
    memoryUse = py.memory_info().rss
    return float(memoryUse)

def get_time():
    return time.time()

def construct_complex_query(size=10):
    alphabet = list(string.ascii_lowercase)
    alphabet.append('')
    query = "/"
    for i in range(size):
        letter = alphabet[random.randint(0, len(alphabet) - 1 )]
        query += "/"  + letter
    return query



def construct_xml(filename = "test_file.xml",max_heigth=10,number_of_node = 100):
    current_height = 0
    total_node = 0
    alphabet = list(string.ascii_lowercase)
    node_added = []
    f  = open(filename,"w")
    for node in range(number_of_node):
        if(current_height < max_heigth  ):
            if(random.randint(0,1) == 0 or not node_added ):
                #create a child node
                letter = alphabet[random.randint(0, len(alphabet) - 1 )]
                f.write("{0}\t{1} \n".format(letter,0))
                node_added.append(letter)
            else:
                # create sibling node
                f.write("{0}\t{1} \n".format(node_added[-1],1))
                node_added.pop()
                letter = alphabet[random.randint(0, len(alphabet) - 1 )]
                f.write("{0}\t{1} \n".format(letter,0))
                node_added.append(letter)
        else:
            f.write("{0}\t{1} \n".format(node_added[-1],1))
            node_added.pop()
            letter = alphabet[random.randint(0, len(alphabet) -  1 )]
            f.write("{0}\t{1} \n".format(letter, 0))
            node_added.append(letter)
    iterations = len(node_added)
    for node in range(iterations):
        f.write("{0}\t{1} \n".format(node_added[-1],1))
        node_added.pop()


def plot_time_mem(title,time_array,memory_array, value_range):
    timestr = time.strftime("%Y%m%d-%H%M%S")
    plt.subplot(211)
    plt.title('Evolution of time with '+ title )
    plt.plot(value_range,time_array)
    ax = plt.subplot(212)
    ax.set_xlabel('Evolution of memory usage with '+ title)
    plt.plot(value_range,memory_array)
    plt.savefig('../data/output/'+title+"-"+timestr+'.png')


def get_perf_lazy(max_heigth=10,number_of_node=100,query_size=10):
    file = "test_file.xml"
    construct_xml(filename = file ,max_heigth=max_heigth,number_of_node = number_of_node)
    query = construct_complex_query(size=query_size)
    path = [query[2:].split('/')]
    begin_memory, begin_time = get_memory_usage(),get_time()
    #print(path, get_failure_transition(path))
    lazy_dfa(file,path)
    end_memory, end_time = get_memory_usage(),get_time()
    #print("time: ", end_time - begin_time, "memory used: ", (end_memory - begin_memory))
    return end_time - begin_time, end_memory - begin_memory


# In[30]:




# In[31]:


# test query_size
query_size_range = range(1,1000)
time_array, memory_array = [],[]
for query_size in query_size_range:
    time_,memory = get_perf_lazy(query_size=query_size)
    time_array.append(time_)
    memory_array.append(memory)
plot_time_mem('query size',time_array,memory_array,query_size_range)


# In[32]:


# test query_size
heigth_range = range(1,10000)
time_array, memory_array = [],[]
for heigth in heigth_range:
    time_,memory = get_perf_lazy(max_heigth=heigth)
    time_array.append(time_)
    memory_array.append(memory)
plot_time_mem('heigth document',time_array,memory_array,heigth_range)



# test query_size
nodes_range = range(1,10000)
time_array, memory_array = [],[]
for nodes in nodes_range:
    time_,memory = get_perf_lazy(number_of_node=nodes)
    time_array.append(time_)
    memory_array.append(memory)
plot_time_mem('the number of node',time_array,memory_array,nodes_range)



nodes_range = range(1,1000)
heigth_range = [10,100,1000,10000]
time_array, memory_array = [],[]
for height in heigth_range:
    time_array, memory_array = [],[]
    plt.figure(height)
    for nodes in nodes_range:
        time_,memory = get_perf_lazy(number_of_node=nodes,max_heigth=heigth)
        time_array.append(time_)
        memory_array.append(memory)
    plot_time_mem('number of node with height:'  + str(height),time_array,memory_array,nodes_range)



nodes_range = range(1,10000)
heigth_range = [10,100,1000,10000,100000]
time_array, memory_array = [],[]
for height in heigth_range:
    time_array, memory_array = [],[]
    plt.figure(height)
    for nodes in nodes_range:
        time_,memory = get_perf_lazy(number_of_node=nodes,max_heigth=heigth)
        time_array.append(time_)
        memory_array.append(memory)
    plot_time_mem('number of node with height:'  + str(height),time_array,memory_array,nodes_range)
