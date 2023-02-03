Steps to reproduce results : 

fastNLP path : home/jivnesh/anaconda3/envs/tlat0/lib/python3.7/site-packages/fastNLP/


conda activate tlat0
cd /media/guest/rathin_workspace/TranLat-main/V0


1.1 Training "SIGHUM Translist Ngram" : 

	- V0/flat_main_bigram.py : change default batch size to 4, i.e. parser.add_argument('--batch', default=4, type=int)
	- In sktWS place SIGHUM data (all 3 datasets are within sktWS)
	- fastNLP/core/dataset.py : new_LREC_data_complete.csv, ngram_lattice_files; in line 808-814 --> choose the correct code for hack_LREC... or new_LREC...
	- use the embeds in the SIGHUM_embeds folder. (just put them in TransLat-main)
	- from V0 directory run : "python flat_main_bigram.py --status train" (in a tmux session)
	- rename saved model in V0/saved_model
	- for testing :
		- V0/flat_main_bigram.py : change the testing model to "best_sighum_ngram2"
		- from V0 run "python flat_main_bigram.py --status test"
		- don't run CI (not applicable in this case)
		Result :
		Accuracy: 86.52380952380952
		Precision: 96.9795516983017
		Recall: 96.77620636638494
		F1_score: 96.87777232758157
	
1.2 Training "SIGHUM Translist SHR"

	- V0/flat_main_bigram.py : change default batch size to 8, i.e. parser.add_argument('--batch', default=8, type=int)
	- In sktWS place SIGHUM data (all 3 datasets are within sktWS)
	- fastNLP/core/dataset.py : new_LREC_data_complete.csv, lattice_files ; in line 808-814 --> choose the correct code for hack_LREC... or new_LREC...
	- use the embeds in the SIGHUM_embeds folder. (just put them in TransLat-main)
	- from V0 directory run : "python flat_main_bigram.py --status train" (in a tmux session)
	- rename saved model in V0/saved_model
	- for testing :
		- V0/flat_main_bigram.py : change the testing model to "best_sighum_shr2"
		- from V0 run "python flat_main_bigram.py --status test"
		- run CI : "python constrained_inference.py" [properly adjust the global paths]
		Result :
			before CI :
			
			Accuracy: 88.85714285714286
			Precision: 97.79255787202216
			Recall: 97.45534871874158
			F1_score: 97.62366210144529
			
			after CI :
			
			Precision: 98.80608455697741
			Recall: 98.93369708994709
			F1_score: 98.8698496457123
			Accuracy: 0.9397619047619048

				

1.3 Training "Hackathon Translist Ngram"

	- V0/flat_main_bigram.py : change default batch size to 4, i.e. parser.add_argument('--batch', default=4, type=int)
	- In sktWS place Hackathon data (all 3 datasets are within sktWS)
	- fastNLP/core/dataset.py : hack_LREC_data_complete.csv, hack_ngram_lattice_files; in line 808-814 --> choose the correct code for hack_LREC... or new_LREC...
	- use the embeds in the Hackathon_data/embeds folder. (just put them in TransLat-main)
	- from V0 directory run : "python flat_main_bigram.py --status train" (in a tmux session)
	- rename saved model in V0/saved_model
	- for testing :
		- V0/flat_main_bigram.py : change the testing model to "best_hack_ngram2"
		- from V0 run "python flat_main_bigram.py --status test"
		- don't run CI (not applicable in this case)
		Result :
		
		Accuracy: 79.28750627195184
		Precision: 96.68802005911002
		Recall: 95.7409579981523
		F1_score: 96.21215848944635

		

1.4 Training "Hackathon Translist SHR"

	- V0/flat_main_bigram.py : change default batch size to 8, i.e. parser.add_argument('--batch', default=8, type=int)
	- In sktWS place Hackathon data (all 3 datasets are within sktWS)
	- fastNLP/core/dataset.py : Hackathon_dcs.csv, hack_shr_lattice_files ; in line 808-814 --> choose the correct code block for hack_LREC... or new_LREC... or Hackathon_dcs
	- use the embeds in the Hackathon_data/embeds folder. (just put them in TransLat-main)
	- from V0 directory run : "python flat_main_bigram.py --status train" (in a tmux session)
	- rename saved model in V0/saved_model
	- for testing :
		- V0/flat_main_bigram.py : change the testing model to "best_hack_shr2"
		- from V0 run "python flat_main_bigram.py --status test"
		- run CI : "python constrained_inference.py" [properly adjust the global paths]
		Result :
		
		before CI :
		
		Accuracy: 78.93627696939288
		Precision: 96.63328072078706
		Recall: 95.69517661239423
		F1_score: 96.16194081140169
		
		after CI :
		
		Precision: 97.78254487232972
		Recall: 97.4412452006605
		F1_score: 97.61159669819875
		Accuracy: 0.8547917711991971






