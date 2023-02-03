import networkx as nx

file_name = 'test_graphmls/435527.graphml'
outfile = 'test_graphmls/created_graphml_435527_hack.txt'

graph_input = open(file_name, mode='rb')
graph = nx.read_graphml(graph_input)
nodes = list(graph.nodes(data = True))


tpls = []
for node in nodes:
	k,d = node
	w = d['word']
	s = d['char_pos']
	e = s + len(w) - 1
	## need to add chunk number also
	ch = d['chunk_no']
	tpls.append((w,s,e,ch))

tpls = sorted(list(set(tpls)))	

with open(outfile,'w') as g:
	for tpl in tpls:
		w,s,e,ch = tpl
		g.write(f"{w},{s},{e},{ch}\n") 
