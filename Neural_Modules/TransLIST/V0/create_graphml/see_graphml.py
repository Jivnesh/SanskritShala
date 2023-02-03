import networkx as nx
import os

def open_file(file_name):
    graph_input = open(file_name, mode='rb')
    graph = nx.read_graphml(graph_input)
    nodes = list(graph.nodes(data = True))
    return nodes

def check_graphml(graphml_file):
# To access the graph details
    graph_abs_file_path = os.path.join(os.getcwd(), graphml_file)
    nodes_ = open_file(graph_abs_file_path)
#    nodes_ = sorted(list(graph_.nodes(data = True)), key = lambda x : x[0])
    #print(len(nodes_))
    nodes = []
    for item in nodes_:
        #print(item)
        first = item[1]
        word = first['word']
        position = first['position']
        chunk_no = first['chunk_no']
        char_pos = first['char_pos']
        length_word = first['length_word']
        nodes.append((char_pos,chunk_no,position,word,length_word))
        #print(f"word : {word}, chunk_no : {chunk_no}, char_pos : {char_pos}")
     
    return nodes
    
    
nodes = sorted(check_graphml('final_graphml_dev/526534.graphml'))
for n in nodes:
	print(n)

