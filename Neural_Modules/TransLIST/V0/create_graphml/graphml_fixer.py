from devconvert import dev2slp, iast2slp, slp2dev, slp2iast, slp2tex, slp2wx, wx2slp, dev2wx

#sent = "yenedaM vyasanaM prAptA Bavanto dyUtakAritam"
#sent = "etac cAnyac ca kOravya prasaNgi kawukodayam"
sent = "corau śvapākacaṇḍālau vipreṇābhihatau yadi"
#sent = "vaiśvānaraḥ śiraḥ pātu viṣṇustava parākramam"
infile = 'test_graphmls/created_graphml_435527_hack.txt'
outfile = 'test_graphmls/corrected_graphml_435527_hack.txt'
form = 'iast'
debug = False

def graphml_iast2slp(sent_slp, triplets_slp):
	
	"""
	in case of sandhi first word will be short, second word would be long. e.g. [0, 2, 'yena'] + [3, 6, 'idam'] = yenedaM {Exception : end of a chunk, end of sentence, sandhi between short and long od samw vowel}
	 
	Find max match in -2 to -2 window
	
	if same match then prefer min deviation
	"""
	
	#triplets = [(t[0],t[1],t[2],t[3]) for t in triplets_slp]
	
	#triplets = list(set(triplets_slp)) ## erase duplicates if any
	
	corrected_triplets = []
	
	for t in triplets_slp:
		if debug : print("\n\ntriplet : ",t)
		w,s,e,c = t
		
		max_pos = 0
		max_match = 0
		my_range = [s,s+1,s-1,s+2,s-2,s+3,s-3,s+4,s-4,s+5,s-5] ## min deviation rule taken care
		if debug : print("my range : ", my_range)
		good_range = [r for r in my_range if r>=0 and r<=len(sent_slp)-1]
		if debug : print("good range : ",good_range)
		for i in good_range:
			k = i
			match = 0
			for j in range(len(w)):
				
				if k>=0 and k<=len(sent_slp)-1 and sent_slp[k] == w[j]: match += 1
				k += 1
			if debug : print("match : ", match)	
			if match > max_match : 
				max_match = match
				max_pos = i
		
		if max_match == 0 : continue
				
		s = max_pos
		e = s + len(w) -1
		if debug : print(sent_slp[e],w[-1])
		if debug : print(s,e)
		dirgha = (sent_slp[e] == w[-1].upper())
		if form == 'iast' : dirgha = (iast2slp.convert(sent_slp[e]) == iast2slp.convert(w[-1]).upper())
		if e == len(sent_slp)-1 or sent_slp[e+1] == ' ' or dirgha : e = e 
		elif sent_slp[e] != w[-1] : e -= 1
		if debug : print(s,e)
		
		corrected_triplets.append((w,s,e,c))
		print(f"{w}, {sent_slp[s:e+1]} ")
				
	return corrected_triplets
	
	

with open(infile,'r') as f:
	lines = f.readlines()
	
triplets = []
for line in lines:
	line = line[:-1]
	sp = line.split(',')
	triplets.append( ( sp[0],int(sp[1]),int(sp[2]),int(sp[3]) ) )
		
new_triplets = graphml_iast2slp(sent,triplets)					
	
with open(outfile,'w') as g:
	for t in new_triplets:
		g.write(str(t)+'\n')
