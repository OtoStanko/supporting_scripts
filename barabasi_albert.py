import matplotlib.pyplot as plt
from networkx import scale_free_graph
from random import shuffle, random
import re
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
rules - list of lists of tuples - [[(node_index, Im, Om)]]
        list of n lists corresponding to n nodes in network
        each inner list consists of tuples
        each tuple represents input node, Im an Om - rule in NCF form
"""

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
part_fun - lis of update rules, each in form of a list of tuples

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
    #print()
    while True:
        new_state = []
        for i in range(n_nodes):  # for each node compute new state
            if i == const:
                new_state.append(value)
                continue
            if rules[i] == []:
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
node_sizes - list/range of sizes of networks to create e.g. [10, 20, 30, 40, 50]
number_of_networks_per_size - number of networks to create of given number of nodes
                              should be as long as node_sizes
"""
file_path = r"D:\MUNI\FI\_bc\supporting_scripts"
# run([5, 11], [1, 2], r"D:\MUNI\FI\_bc\supporting_scripts")
def run(node_sizes, number_of_networks_per_size, output_directory_path=None):

    if len(node_sizes) != len(number_of_networks_per_size):
        print("node sizes and numbers of networks is not equally long")
        return

    for i in range(len(node_sizes)):
        size = node_sizes[i]
        number_of_networks = number_of_networks_per_size[i]
        for j in range(number_of_networks):
            file_path = output_directory_path + "\\size" + str(size) + "_n" + str(j)
            #print(file_path)
            H, rules = create_scale_free_network_with_rules(size, file_path)

            funs = create_update_function_stm(rules)
            other_funs = create_update_functions_from_rules(rules)

            n = [Bool("n" + str(i)) for i in range(H.number_of_nodes())]

            initial_state = find_steady_state(funs, n)
            if initial_state is None:
                return
            matrix = generate_steady_state_matrix(initial_state, rules)
            write_graph_to_file(funs, file_path + "original_network.txt")
            write_matrix_to_file(matrix, file_path + "_output_matrix.csv")
    # end of run()


def create_scale_free_network_with_rules(number_of_nodes, file_path):
    m=2
    p=0.6

    # generate undirected scale-free graph using barabasi-albert model
    # n - number of nodes on the network
    # m1 - number of edges added from every new node with P = p
    # m2 - number of edges added from every new node with P = 1-p
    """G = nx.dual_barabasi_albert_graph(n=number_of_nodes,
                                      m1=m-1, m2=m, p=p,
                                      initial_graph=nx.complete_graph(m+2))"""
    H = scale_free_graph(number_of_nodes, alpha=0.41,
                         beta=0.54, gamma=0.05, delta_in=0.2,
                         delta_out=0, create_using=None,
                         seed=None)

    """# create empty directed graph
    H = nx.DiGraph()
    # add all nodes from undirected graph to the new one
    H.add_nodes_from(G)"""

    # create table of update rules represented as list of lists of tuples
    rules = [[] for _ in range(H.number_of_nodes())]  # [[(node_index, Im, Om)]]

    # for each edge before putting it into the directed graph
    #    randomly choose direction of the new edge
    # specify Im and Om to the update rules -> randomly generate Im and Om
    for edge in H.edges():
        s, t = edge
        Im = int(2*random())
        Om = int(2*random())
        rules[t].append((s, Im, Om))
    """for edge in G.edges():
        s, t = edge
        Im = int(2*rndm.random())
        Om = int(2*rndm.random())
        if rndm.random() < 0.5:
            H.add_edge(s, t)
            rules[t].append((s, Im, Om))
        else:
            H.add_edge(t, s)
            rules[s].append((t, Im, Om))"""
    shuffle_argumentsOfUpdate_rules(rules)
    return H, rules


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
        model = s.model()  # model - steady state
        for node in n:
            initial_state.append(model[node])
            #print(str(node) + " = " + str(model[node]))
    else:
        print("Steady state does not exist")
        return None
    initial_state = [1 if is_true(i) else 0 for i in initial_state]
    
    return initial_state


# nx.draw(H, with_labels=True)
# plt.show()
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
            #print(simulation_path)
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
            print(str(index) + ", " + str(line)[1:-1], file=f)


def write_graph_to_file(funs, file_path):
    with open(file_path, "w") as f:
        for fun in funs:
            neg = re.findall(r'Not\(n\[(\d*)\]\)', fun)
            pos = re.findall(r'n\[(\d*)\]', fun)
            tg = pos[0]
            for sg in neg:
                print(str(sg) + "   " + str(tg) + " " + "-", file=f)
            for sg in pos[1:]:
                if sg not in neg:
                    print(str(sg) + "   " + str(tg) + " " + "+", file=f)
            
# cycle = simulation(rules, tuple_initial_state)[-1]
# print(len(cycle), cycle)
# print(tuple(initial_state) == cycle[0])
                      
