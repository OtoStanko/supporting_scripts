import os
import pandas as pd
import matplotlib.pyplot as plt
from numpy import std


global network_times, iters_time, iters_num, states_req, states_gen, states_met, states_tim, networks

networks = []
network_time = "networkTime"
sum_time = "sumTime"  # time of the whole program (execute function)
iter_time = "Finished searching"
iter_results = "GA_searchBN_results"
states_results = "[States]"
states_time = "[States_time]"
constraints = "computing constraints"

keyWords = [network_time, sum_time, iter_time,
            iter_results, states_results,
            constraints]

network_times = []  # time needed for the whole network computation
iters_time = []  # time needed for the simulation
iters_num = []  # number of iterations in simulation

states_req = []  # num of required states
states_gen = []  # num of actually generated states
states_met = []  # method used for states generation
states_tim = []  # time to generate states

const_time = []  # time for computing the constraints

results = {"network_time": network_times,
           "iter_time": iters_time,
           "iter_num": iters_num,
           "required_states": states_req,
           "generated_states": states_gen,
           "method_of_generation": states_met,
           "constraints_time": const_time}

def add_time(line):
    network_times.append(int(line.split("_")[1]))

def get_final_time(line):
    return int(line.split("_")[1])

def add_iter_time(line):
    iters_time.append(int(line.split(" ")[-2]))

def add_iters_num(line):
    iters_num.append(int(line.split(" ")[1].split(",")[0].split("_")[1]))

def add_states_info(line):
    act, req, met = line.split(" ")[1].split(",")
    states_req.append(int(req.split("_")[1]))
    states_gen.append(int(act.split("_")[1]))
    states_met.append(int(met.split("_")[1]))

def add_states_time(line):
    states_tim.append(line.split("_")[2])

def add_constraints(line):
    const_time.append(int(line.split(" ")[-1]))

def parse_line(line):
    global network_times, iters_time, iters_num, states_req, states_gen, states_met, states_tim, networks
    if "KO File" in line:
        networks.append((network_times, iters_time,
                         iters_num, states_req,
                         states_gen, states_met,
                         states_tim))
        network_times = []
        iters_time = []
        iters_num = []
        states_req = []
        states_gen = []
        states_met = []
        states_tim = []
    elif network_time in line:
        add_time(line)
    elif sum_time in line:
        print(get_final_time(line))
    elif iter_time in line:
        add_iter_time(line)
    elif iter_results in line:
        add_iters_num(line)
    elif states_results in line:
        add_states_info(line)
    elif states_time in line:
        add_states_time(line)
    elif constraints in line:
        add_constraints(line)


def plot_results():
    plotdata = pd.DataFrame({
        "iters": iters_time,
        "other": [x-y for x,y in zip(network_times, iters_time)]
    },
    index=["run"+str(x) for x in range(1, len(states_req)+1)])
    plotdata.plot(kind='bar', stacked=True, figsize=(15, 8))

    plt.title("Time of the algorithm")
    plt.xlabel("run number")
    plt.ylabel("time in ms")
    plt.show()



# main(r"D:\MUNI\FI\_bc\genericalgorithmbackup\CGAProj\CGA-BNI-main\data\outputs.txt")
def main(output_file_path):
    global network_times, iters_time, iters_num, states_req, states_gen, states_met, states_tim, networks
    networks = []
    network_times = []
    iters_time = []
    iters_num = []
    states_req = []
    states_gen = []
    states_met = []
    states_tim = []
    with open(output_file_path, "r") as f:
        for line in f:
            parse_line(line)
        parse_line("KO File")
        networks.remove(([], [], [], [], [], [], []))
    #for key, value in results.items():
        #print(key, value)
    #plot_results()


def print_states_results_all():
    for i in range(24, 33):
        get_initial_states_results("_"+str(1000*i))
    get_initial_states_results("_all")
    print("************************")
    for i in range(24, 33):
        get_initial_states_results("_new_"+str(1000*i))
    get_initial_states_results("_new_all")


def get_initial_states_results(sufix):
    main(r"D:\MUNI\FI\_bc\genericalgorithmbackup\CGAProj\CGA-BNI-main\data\outputs" + sufix + ".txt")
    networks_times_sum = []
    states_sum = []
    for network in networks:
        networks_times_sum.append(sum([ int(x[:-1]) for x in network[6]]))
        states_sum.append(sum(network[4])/len(network[4]))
    print("STATES required:", sufix)
    print("TIME mean: ", sum(networks_times_sum) / len(networks_times_sum))
    print("TIME std:", std(networks_times_sum))
    print("STATES mean:", sum(states_sum) / len(states_sum))
    print("STATES std:", std(states_sum))
    print("-----------------------------------")


def plot_states():
    required = [1000*x for x in range(24, 33)]
    required.append(32768)
    # States
    avg_states_old = [43226, 47163, 51719, 56875, 63111, 70950, 81000, 95724, 122821, 349643]
    avg_states_new = [42980, 41635, 40347, 39104, 37916, 36768, 35659, 34585, 33545, 32768]
    plt.plot(required, avg_states_old, label="Original method")
    plt.plot(required, avg_states_new, label="New method")

    plt.xlabel("Required states")
    plt.ylabel("Number of states")
    plt.legend()
    plt.show()


def plot_times():
    required = [1000*x for x in range(24, 33)]
    required.append(32768)
    # Time
    total_time_old = [1563, 1524, 1667, 1839, 2005, 2240, 2740, 3101, 3926, 11071]
    total_time_new = [2744, 2386, 2312, 2286, 2248, 2216, 2427, 2128, 2079, 2158]
    plt.plot(required, total_time_old, label="Original method")
    plt.plot(required, total_time_new, label="New method")

    plt.xlabel("Required states")
    plt.ylabel("Time (ms)")
    plt.legend()
    plt.show()


def sim_time_per_iter():
    for i in range(len(networks)):
        network = networks[i]
        if i % 20 == 0:
            print("******************************")
        print("times:", network[1][0], network[1][1])
        print("iters:", network[2][0], network[2][0])
        print(sum(network[1])/sum(network[2]))

def sim_time_per_iter_per_n(sufix=""):
    main(r"D:\MUNI\FI\_bc\genericalgorithmbackup\CGAProj\CGA-BNI-main\data\outputs" +sufix+ ".txt")
    sum_times = 0
    sum_iters = 0
    all_times = []
    for i in range(len(networks)):
        network = networks[i]
        all_times = all_times + network[1]
        sum_times += sum(network[1])
        sum_iters += sum(network[2])
    print(sum_times/sum_iters)
    print(std(all_times)/sum_iters)

def plot_iters():
    required = [10000, 20000]
    # Time
    total_time_old = [1563, 1524, 1667, 1839, 2005, 2240, 2740, 3101, 3926, 11071]
    total_time_new = [2744, 2386, 2312, 2286, 2248, 2216, 2427, 2128, 2079, 2158]
    plt.plot(required, total_time_old, label="Original method")
    plt.plot(required, total_time_new, label="New method")

    plt.xlabel("Required states")
    plt.ylabel("Time (ms)")
    plt.legend()
    plt.show()
