import pandas

df = pandas.read_csv("../LREC-Data/new_LREC_data_complete.csv")
extra_conn_inputs = df['input'].tolist()
extra_conn_dcs = df['DCS-ID'].tolist()

#sent = "yenedaM vyasanaM prAptA Bavanto dyUtakAritam"
sent = "etac cAnyac ca kOravya prasaNgi kawukodayam"
print(sent)

dcs_id_str = str(extra_conn_dcs[extra_conn_inputs.index(sent)])

path_to_lattice = '../lattice_files/'+dcs_id_str+'.lat'

add_conn = pandas.read_csv(path_to_lattice)
                        
temp1 = add_conn.values.tolist()

for t in temp1:
	print(t)
