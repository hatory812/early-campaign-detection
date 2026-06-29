import glob
import json
import sys
import math
import numpy as np
import argparse
import random
from random import sample
import networkx as nx
from networkx.readwrite import json_graph
from gensim.models.word2vec import Word2Vec
from k_truss import get_k_truss, load_k_truss

def convert_file_id_to_name(path, id):
    global tmp_cnt
    graph_id_map = json.load(
        open('./graph_name_mapping.json'))
    id_graph_map = dict((v, k) for k, v in graph_id_map.items())
    fname = id_graph_map[int(id)]
    file_path = path+"/"+fname.rstrip()+'_fulldata.json'

    return file_path

def remove_zn(graph, neighbors):
    zn = []

    #degrees = []
    for n in neighbors:
        #degrees.append(graph.degree(n))
        if graph.degree(n) == 0:
            zn.append(n)

    for n in zn:
        neighbors.remove(n)

    return neighbors

def get_rww(graph, pick, comp_parameter):
    u_graph = graph.to_undirected()
    u_graph.remove_edges_from(nx.selfloop_edges(u_graph))

    walk_length = 100
    num_walks = 1

    random_walks = []

    if pick == 'kcore':
        attribute='kcore_value'
    elif pick == 'ktruss':
        attribute='ktruss_value'
    else:
        attribute = 'degree_value'

    attributes = nx.get_node_attributes(u_graph, attribute)
    attr_values = list(attributes.values())

    comp = 0.5
    # Comment the right comperator
    if comp_parameter == 'median':
        comp = np.median(attr_values)
    elif comp_parameter == 'mid':
        comp = (min(attr_values) + max(attr_values)) / 2


    for node in u_graph.nodes():
        for w in range(num_walks):
            walk = [node]

            if u_graph.degree(node):
                for l in range(walk_length):
                    neighbors = remove_zn(u_graph, list(u_graph.neighbors(walk[-1])))
                    if len(neighbors):
                        if u_graph.nodes[walk[-1]][attribute] > comp:
                            weights = np.array([u_graph.nodes[nbr][attribute] for nbr in neighbors])
                        else:
                            weights = np.array([1-u_graph.nodes[nbr][attribute] for nbr in neighbors])

                        # Min-max normalization can yield all-zero weights for a
                        # neighborhood; fall back to a uniform distribution then.
                        if weights.sum() == 0:
                            weights = np.ones(len(neighbors))

                        next_node = random.choices(neighbors, weights=weights, k=1)
                        walk.append(next_node[0])
                random_walks.append(walk)
            else:
                random_walks.append(walk)

    return random_walks

def get_core_numbers(graph):
    u_graph = graph.to_undirected()
    u_graph.remove_edges_from(nx.selfloop_edges(u_graph))

    kcore = list(nx.core_number(u_graph).values())
    # Min-max normalize so density values fall in [0, 1] and are comparable
    # to the threshold tau used in Algorithm 1 (DECODE).
    kmin, kmax = min(kcore), max(kcore)
    if kmax > kmin:
        kcore = [float((x - kmin) / (kmax - kmin)) for x in kcore]
    else:
        kcore = [0.0 for _ in kcore]

    for idx, node in enumerate(u_graph.nodes()):
        graph.nodes[node]['kcore_value'] = kcore[idx]

    return graph

def get_truss_numbers(graph, graph_name):
    ktruss = load_k_truss(graph_name, graph.edges())
    node_weights = {node: [] for node in graph.nodes()}

    edges = list(graph.edges())
    for i in range(len(edges)):
        u, v = edges[i]
        ktruss_value = ktruss[i]  # Assuming ktruss is a dict with edge tuples as keys
        # Append the ktruss value to both nodes (for in-degree and out-degree)
        node_weights[u].append(ktruss_value)
        node_weights[v].append(ktruss_value)

    truss_weights = []
    for node, values in node_weights.items():
        if values:
            truss_weights.append(sum(values) / len(values))
        else:
            truss_weights.append(0)

    # Min-max normalize so density values fall in [0, 1] and are comparable
    # to the threshold tau used in Algorithm 1 (DECODE).
    tmin, tmax = min(truss_weights), max(truss_weights)
    if tmax > tmin:
        truss_weights = [float((x - tmin) / (tmax - tmin)) for x in truss_weights]
    else:
        truss_weights = [0.0 for _ in truss_weights]

    for idx, node in enumerate(graph.nodes()):
        graph.nodes[node]['ktruss_value'] = truss_weights[idx]

    return graph

def get_degree(graph):
    u_graph = graph.to_undirected()
    u_graph.remove_edges_from(nx.selfloop_edges(u_graph))

    # Compute the degree of each node and apply the same exponential transformation

    degree = [u_graph.degree(node) for node in u_graph.nodes()]
    # Min-max normalize so density values fall in [0, 1] and are comparable
    # to the threshold tau used in Algorithm 1 (DECODE).
    dmin, dmax = min(degree), max(degree)
    if dmax > dmin:
        degree = [float((x - dmin) / (dmax - dmin)) for x in degree]
    else:
        degree = [0.0 for _ in degree]

    for idx, node in enumerate(u_graph.nodes()):
        graph.nodes[node]['degree_value'] = degree[idx]

    return graph

def get_embedding(walks, graph, pick):
    model = Word2Vec(
        walks,
        hs=1,
        alpha=0.0001,
        epochs=100,
        vector_size=128,
        window=5,
        min_count=1,
        workers=4,
        seed=42,
    )
    embedding = []

    for n in range(graph.number_of_nodes()):
        if pick == 'kcore':
            graph.nodes[n]['structural_embedding'] = model.wv[n].tolist()
        elif pick == 'ktruss':
            graph.nodes[n]['truss_embedding'] = model.wv[n].tolist()
        elif pick == 'deepwalk':
            graph.nodes[n]['deepwalk_embedding'] = model.wv[n].tolist()
        else:
            graph.nodes[n]['degree_embedding'] = model.wv[n].tolist()


    return graph

def load_data(file, file_name, pick):
    with open(file, 'r') as f:
        data = json.load(f)
    graph = nx.DiGraph(json_graph.node_link_graph(data))
    mapping = {node: i for i, node in enumerate(graph.nodes())}
    graph = nx.relabel.relabel_nodes(graph, mapping)

    if(pick=="kcore"):
        graph = get_core_numbers(graph)
    elif (pick == "ktruss"):
        graph = get_truss_numbers(graph, file_name)
    else:
        graph = get_degree(graph)

    return graph

def run_model(file, path, pick, comp_parameter):
    file_name = file.split('/')[-1][:-5]

    print(file_name)
    with open(path + "/graph_labels.json", "r") as f:
        graph_labels = json.load(f)
    if (file_name not in ['graph_labels']
            and (file_name[:-9] in graph_labels)):
        # 1. build the basic graph with kcore values
        graph = load_data(file, file_name, pick)

        # 2. Perform random walks
        walks = get_rww(graph, pick, comp_parameter)

        graph = get_embedding(walks, graph, pick)
        graph_json = nx.node_link_data(graph)

        output_path = f"/vscratch/grp-erdem/atulanan/graphs/{pick}/{comp_parameter}"

        file_name = output_path + "/" + file_name + '.json'
        with open(file_name, 'w') as f:
            json.dump(graph_json, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--graphId', type=int, help="Enter file is", default=0)
    parser.add_argument('--pick', help="Degree or K-Core", default="kcore")
    parser.add_argument("--comp", help="Parameter for comparison", default ="0.5")
    parser.add_argument("--path", help="Path where graphs are stored", default="./")

    args = parser.parse_args()

    path = args.path
    comp_parameter = args.comp

    try:
        filename = convert_file_id_to_name(path, args.graphId)
    except KeyError:
        print(f"Skipping graphId {args.graphId}: not found in graph_name_mapping.json")
        sys.exit(0)

    run_model(filename, path, args.pick, comp_parameter)