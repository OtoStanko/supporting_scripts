import matplotlib.pyplot as plt
from networkx import scale_free_graph, draw, dual_barabasi_albert_graph, complete_graph, DiGraph
from random import shuffle, random
import re
import sys
#import converters as conv
#from sympy import symbols, true, false
#from sympy.logic import And, Or, Not, simplify_logic
from z3 import *


"""
example set of update functions
"""
'''
exa_funs = ["n0 = n1 & (n3 | (n4 & n8))", "n1 = ~ n2", "n2 = ~ n0", "n3 = n1 & ~ n2",
            "n4 = n5 & n9", "n5 =", "n6 = ~ n1", "n7 = n1", "n8 = ~ n1", "n9 ="]

exa_funs_z = ["n[0] == And(n[1], Or(n[3], And(n[4], n[8])))", "n[1] == Not(n[2])", "n[2] == Not(n[0])",
              "n[3] == And(n[1], Not(n[2]))", "n[4] == And(n[5], n[9])", "n[5] ==",
              "n[6] == Not(n[1])", "n[7] == n[1]", "n[8] == Not(n[1])", "n[9] =="]

n = [Bool("n0"), Bool("n1"), Bool("n2"), Bool("n3"), Bool("n4"),
     Bool("n5"), Bool("n6"), Bool("n7"), Bool("n8"), Bool("n9")]
"""
strongly connected components and how the network should look like after simplification
input nodes = 5, 9
"""
exa_strong_components = [{9}, {5}, {4}, {0, 1, 2, 3, 8}, {6}, {7}]
exa_funs_simple = ["A = A", "B = A", "C = ~A", "D = A",
                   "E = F & J", "F =", "G = ~A", "H = A", "I = ~A", "J ="]
exa_input_nodes = [5, 9]
exa_output_nodes = [6, 7]'''



"""
function - string representing update function of a node
           all of the missing right brackets are appended
"""
def close_function(function: str):
    if function[-1] == "=":
        return function
    index_of_last_digit = 0
    i = 0
    number_of_left_brackets = 0
    number_of_right_brackets = 0
    for char in function:
        if char.isnumeric():
            index_of_last_digit = i
        elif char == '(':
            number_of_left_brackets += 1
        elif char == ')':
            number_of_right_brackets += 1
        i += 1
    new_function = function[:index_of_last_digit+2] + ")"*(number_of_left_brackets-number_of_right_brackets-1)
    return new_function


"""
part_fun - update rule in form of a list of tuples

return string representing update function in a form that sat z3-solver can process
"""
def build_up_fun_rec(part_fun):
    if part_fun == []:
        return ""
    if len(part_fun) == 1:
        neg1 = "Not(" * (part_fun[0][1] != part_fun[0][2])
        neg2 = ")" * (neg1 != "")
        return " " + neg1 + "n[" + str(part_fun[0][0]) + "]" + neg2
    node_index, Im, Om = part_fun[0]
    if Im == Om:
        if Im == 0:
            return " And(n[" + str(node_index) + "], " + build_up_fun_rec(part_fun[1:]) + ")"
        else:
            return " Or(n[" + str(node_index) + "], " + build_up_fun_rec(part_fun[1:]) + ")"
    else:
        if Im == 0:
            return " Or( Not(n[" + str(node_index) + "]), " + build_up_fun_rec(part_fun[1:]) + ")"
        else:
            return " And( Not(n[" + str(node_index) + "]), " + build_up_fun_rec(part_fun[1:]) + ")"


"""
rules - list of lists of tuples - [[(node_index, Im, Om)]]
        list of n lists corresponding to n nodes in network
        each inner list consists of tuples
        each tuple represents input node, Im an Om - rule in NCF form

returns list of strings representing update functions in a form that sat z3-solver can process
"""
def create_update_function_stm(rules):
    functions = []
    counter = 0
    for node_table in rules:
        function = "n[" + str(counter) + "] ==" + build_up_fun_rec(node_table)
        functions.append(function)
        counter += 1
    return functions


def create_update_functions_from_rules(rules):
    """
    Im, Om
    0, 0 - X AND
    0, 1 - NOT X OR
    1, 0 - NOT X AND
    1, 1 - X OR
    """
    functions = []
    counter = 0
    for node_table in rules:
        function = "n" + str(counter) + " =="
        for node_index, Im, Om in node_table:
            if Im == Om:
                if Im == 0:
                    function = function + " n" + str(node_index) + " & ("
                else:
                    function = function + " n" + str(node_index) + " | ("
            else:
                if Im == 0:
                    function = function + " ~ (n" + str(node_index) + ") | ("
                else:
                    function = function + " ~ (n" + str(node_index) + ") & ("
        functions.append(close_function(function))
        counter += 1
    return functions


def shuffle_argumentsOfUpdate_rules(update_rules):
    for i in range(len(update_rules)):
        shuffle(update_rules[i])


def simulation(rules, initial_state, const=None, value=None):
    visited = [initial_state]
    n_nodes = len(rules)
    actual_state = initial_state
    while True:
        new_state = []
        for i in range(n_nodes):  # for each node compute new state
            if i == const:
                new_state.append(value)
                continue
            if not rules[i]:
                new_state.append(actual_state[i])
                continue
            flag = False
            for node, Im, Om in rules[i]:
                if Im == actual_state[node]:
                    new_state.append(1*(Om == True))
                    flag = True
                    break
            if not flag:
                new_state.append( 1*((1 - rules[i][-1][2]) == True) )
        new_state_tuple = tuple(new_state)
        if new_state_tuple in visited:
            visited.append(new_state_tuple)
            return visited
        visited.append(new_state_tuple)
        actual_state = new_state_tuple


"""
            Create random network with random update functions
"""


"""
network_sizes - list/range of sizes of networks to create e.g. [10, 20, 30, 40, 50]
number_of_networks_per_size - number of networks to create of given number of nodes
                              should be as long as network_sizes
"""
file_path = r"D:\MUNI\FI\_bc\supporting_scripts"
# run([5, 11], [1, 2], r"D:\MUNI\FI\_bc\supporting_scripts")
# 5,11 1,2 D:\\MUNI\\FI\\_bc\\supporting_scripts
# 50,100,150,200 25,25,25,25 D:\\MUNI\\FI\\_bc\\supporting_scripts
# 10,20 5,5 D:\\MUNI\\FI\\_bc\\supporting_scripts
# 15 40 D:\\MUNI\\FI\\_bc\\supporting_scripts
# 100 1 D:\\MUNI\\FI\\_bc\\supporting_scripts
def run(network_sizes, number_of_networks_per_size, output_directory_path=None):

    if len(network_sizes) != len(number_of_networks_per_size):
        print("node sizes and numbers of networks is not equally long")
        return 1

    for i in range(len(network_sizes)):
        size = network_sizes[i]
        number_of_networks = number_of_networks_per_size[i]

        size_dir_path = output_directory_path + "\\Size" + str(size)
        if not os.path.isdir(size_dir_path):
            os.mkdir(size_dir_path)
        method = "scf"
        for j in range(number_of_networks):
            if j >= number_of_networks:
                method = "ba"
            file_path = size_dir_path + "\\Size" + str(size) +\
                        "_RBN" + str(j)
            H = create_scale_free_network(size, method)
            rules = generate_rules(H)
            n = [Bool("n" + str(i)) for i in range(H.number_of_nodes())]

            funs = create_update_function_stm(rules)
            other_funs = create_update_functions_from_rules(rules)


            initial_state = find_steady_state(funs, n)
            if initial_state is None:
                continue
            matrix = generate_steady_state_matrix(initial_state, rules)
            write_graph_to_file(funs, file_path + "_goldstandard_signed.tsv")
            write_matrix_to_file(matrix, file_path + "_knockouts.tsv.bool.csv")
    return 0
    # end of run()

"""
number_of_nodes - number of nodes in the generated network
method          - method used for generation of scale-free network
                    scf - basic scale_free_graph - produces directed graph
                    ba  - Barabasi-Albert model  - produces undirected graph
                                                   for each edge, direction
                                                   is randomly chosen
"""
def create_scale_free_network(number_of_nodes, method="scf"):
    if method == "scf":
        # generate directed scale-free graph
        H = scale_free_graph(number_of_nodes, alpha=0.05,
                             beta=0.35, gamma=0.6, delta_in=0.2,
                             delta_out=0, create_using=None,
                             seed=None)
        # above function generates duplicate edges as well as self-loops
        # these need to be removed
        edges_count = {}
        for edge in H.edges():
            edges_count[edge] = edges_count.get(edge, 0) + 1
        for edge, n in edges_count.items():
            s, t = edge
            for _ in range(n-1):
                H.remove_edge(s, t)
            if s == t:
                H.remove_edge(s, t)
    elif method == "ba":
        m=2
        G = dual_barabasi_albert_graph(number_of_nodes, m1=m, m2=1, p=0.6,
                                 initial_graph=complete_graph(m+2))
        H = DiGraph()
        H.add_nodes_from(G)
        for edge in G.edges():
            s, t = edge
        if random() < 0.5:
            H.add_edge(s, t)
        else:
            H.add_edge(t, s)
    return H


def generate_rules(H):
    # create table of update rules represented as list of lists of tuples
    rules = [[] for _ in range(H.number_of_nodes())]  # [[(node_index, Im, Om)]]
    # specify Im and Om to the update rules -> randomly generate Im and Om
    for edge in H.edges():
        s, t = edge
        Im = int(2*random())
        Om = int(2*random())
        rules[t].append((s, Im, Om))
    shuffle_argumentsOfUpdate_rules(rules)
    # draw(H, with_labels=True)
    # plt.show()
    return rules


"""
            Find steady-state
"""
def find_steady_state(funs, n):
    initial_state = []
    s = Solver()
    for fun in funs:
        if fun[-1] != "=":
            s.add(eval(fun))
    if s.check() == sat:
        model = s.model()  # model = steady state
        for node in n:
            initial_state.append(model[node])
            #print(str(node) + " = " + str(model[node]))
    else:
        print("Steady state does not exist")
        return None
    initial_state = [1 if is_true(i) else 0 for i in initial_state]
    
    return initial_state


"""
            Generate steady-state matrix
"""
def generate_steady_state_matrix(initial_state, rules):
    tuple_initial_state = tuple(initial_state)
    matrix = [ (-1, tuple_initial_state) ]
    if initial_state != []:
        for i in range(len(initial_state)):
            simulation_path = simulation(rules,
                                         tuple_initial_state,
                                         i,
                                         1*(1-initial_state[i]))
            if simulation_path[-1] == simulation_path[-2]:
                matrix.append( (i, simulation_path[-1]) )
            else:
                print("There exists cycle, but not a sink")
    return matrix


"""
            write matrix and graph to the files
"""
def write_matrix_to_file(matrix, file_path):
    with open(file_path, "w") as f:
        for index, line in matrix:
            if index == -1:
                print(str(index) + ",0," + str(line)[1:-1].replace(" ", ""), file=f)
            else:
                print(plus1(index) + ",0," + str(line)[1:-1].replace(" ", ""), file=f)


def plus1(n):
    return str(int(n)+1)


# writing graph's topology to the file given by path
def write_graph_to_file(funs, file_path):
    space = "\t"
    with open(file_path, "w") as f:
        for fun in funs:
            neg = re.findall(r'Not\(n\[(\d*)\]\)', fun)
            pos = re.findall(r'n\[(\d*)\]', fun)
            tg = pos[0]
            for sg in neg:
                print("G" + plus1(sg) + space + "G" + plus1(tg) + space + "-", file=f)
            for sg in pos[1:]:
                if sg not in neg:
                    print("G" + plus1(sg) + space + "G" + plus1(tg) + space + "+", file=f)
            
# cycle = simulation(rules, tuple_initial_state)[-1]
# print(len(cycle), cycle)
# print(tuple(initial_state) == cycle[0])
def main():
    args = sys.argv
    print(args)
    if len(args) != 4:
        print("wrong number of arguments:", len(args)-1, "expected 3")
        return 1
    network_sizes = [int(i) for i in args[1].split(",")]
    number_of_networks_per_size = [int(i) for i in args[2].split(",")]
    output_directory_path = args[3]
    return run(network_sizes, number_of_networks_per_size, output_directory_path)
    


if __name__=='__main__':
    sys.exit(main())
    
