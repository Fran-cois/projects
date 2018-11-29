
from lazy_dfa import *
from linear_parsing import *
def get_memory_usage():
    pid = os.getpid()
    py = psutil.Process(pid)
    #memoryUse = py.memory_info()[0]/2.**36  # memory use in kB
    memoryUse = py.memory_info().rss
    return float(memoryUse)

def get_time():
    return time.time()

src = '../data/'

class TestLinear(unittest.TestCase):
    def test_equals(self):
        file = src + 'officials_examples/input.txt'
        queries = ["//a","//b","//c","//a/b","//a/a"]
        paths = [query[2:].split('/') for query in queries]
        linear_test_paths = paths
        VERBOSE = False
        expected_results = [
            [0, 2, 3, 4, 8, 9, 11, 12, 15],
            [1, 5, 6, 10, 13, 14, 16],
            [7],
            [1, 5, 6, 10, 13, 14, 16],
            [3, 4, 8, 9, 11, 12]
        ]
        i = 0
        for path in linear_test_paths:
            #print("results",linear_parse(file,path))
            begin_memory, begin_time = get_memory_usage(),get_time()
            print(path, get_failure_transition(path))
            memory  = 0
            for name, obj in locals().items():
                memory += sys.getsizeof(obj)
            print(memory)
            self.assertSequenceEqual(linear_parse(file,path),expected_results[i])
            i += 1
            end_memory, end_time = get_memory_usage(),get_time()
            print("memory, before : ", begin_memory," after: ", end_memory)
            memory  = 0
            for name, obj in locals().items():
                memory += sys.getsizeof(obj)
            print(memory)
            print("time: ", end_time - begin_time, "memory used: ", (end_memory - begin_memory))


class Testlazy(unittest.TestCase):
    def test_equals_2(self):
        file = src + 'officials_examples/input.txt'
        queries = ["//a","//b","//c","//a/b","//a/a","//a/b//a","//a/b//a/b"]
        paths = [query[2:].split('/') for query in queries]
        test_paths = paths
        VERBOSE = False
        expected_results = [
            [0, 2, 3, 4, 8, 9, 11, 12, 15],
            [1, 5, 6, 10, 13, 14, 16],
            [7],
            [1, 5, 6, 10, 13, 14, 16],
            [3, 4, 8, 9, 11, 12],
            [2, 3, 4, 15],
            [5, 6, 16]
        ]
        i = 0
        for path in test_paths:
            #DEBUG_print("results",linear_parse(file,path))
            begin_memory, begin_time = get_memory_usage(),get_time()
            #print(path, get_failure_transition(path))
            memory  = 0
            for name, obj in locals().items():
                memory += sys.getsizeof(obj)
            #print(memory)
            #print(lazy_dfa(file,path),expected_results[i])
            self.assertSequenceEqual(lazy_dfa(file,path),expected_results[i])
            i += 1
            end_memory, end_time = get_memory_usage(),get_time()
            print("memory, before : ", begin_memory," after: ", end_memory)
            memory  = 0
            for name, obj in locals().items():
                memory += sys.getsizeof(obj)
            print(memory)
            print("time: ", end_time - begin_time, "memory used: ", (end_memory - begin_memory))


if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
