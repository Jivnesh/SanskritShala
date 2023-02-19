
domain="san"

word_path="./data/multilingual_word_embeddings/cc.sanskrit.300.new.vec" 
declare -i num_epochs=100
declare -i word_dim=300
declare -i set_num_training_samples=500
start_time=`date +%s`
current_time=$(date "+%Y.%m.%d-%H.%M.%S")
model_path="ud_parser_san_"$current_time 


echo "#################################################################"
echo "Currently BiAFFINE model training in progress..."
echo "#################################################################"
python examples/GraphParser.py --dataset ud --domain $domain --rnn_mode LSTM \
--num_epochs $num_epochs --batch_size 16 --hidden_size 512 --arc_space 512 \
--arc_tag_space 128 --num_layers 2 --num_filters 100 --use_char \
--set_num_training_samples $set_num_training_samples \
--word_dim $word_dim --char_dim 100 --pos_dim 100 --initializer xavier --opt adam \
--learning_rate 0.002 --decay_rate 0.5 --schedule 6 --clip 5.0 --gamma 0.0 \
--epsilon 1e-6 --p_rnn 0.33 0.33 --p_in 0.33 --p_out 0.33 --arc_decode mst \
--punct_set '.' '``'  ':' ','  --word_embedding fasttext --char_embedding random  --pos_embedding random --word_path $word_path \
--model_path saved_models/$model_path 2>&1 | tee saved_models/base_log.txt

mv saved_models/base_log.txt saved_models/$model_path/base_log.txt

##################################################################
## Pretrainig Step
## For DCST setting set tasks as : 'number_of_children' 'relative_pos_based' 'distance_from_the_root'
## For LCM set tasks as: 'Multitask_case_predict' 'Multitask_POS_predict' 'add_label'
for task in 'Multitask_case_predict' 'Multitask_POS_predict' 'add_label'; do
	touch saved_models/$model_path/log.txt
	echo "#################################################################"
	echo "Currently $task in progress..."
	echo "#################################################################"  
    python examples/SequenceTagger.py --dataset ud --domain $domain --task $task \
    --rnn_mode LSTM --num_epochs $num_epochs --batch_size 16 --hidden_size 512 \
	--tag_space 128 --num_layers 2 --num_filters 100 --use_char --char_dim 100 \
	--pos_dim 100 --initializer xavier --opt adam --learning_rate 0.002 --decay_rate 0.5 \
	--schedule 6 --clip 5.0 --gamma 0.0 --epsilon 1e-6 --p_rnn 0.33 0.33 \
	--p_in 0.33 --p_out 0.33 --punct_set '.' '``'  ':' ','  \
	--word_dim $word_dim --word_embedding fasttext --word_path $word_path --pos_embedding random \
	--parser_path saved_models/$model_path/ \
	--use_unlabeled_data --use_labeled_data --char_embedding random \
	--model_path saved_models/$model_path/$task/ 2>&1 | tee saved_models/$model_path/log.txt
	mv saved_models/$model_path/log.txt saved_models/$model_path/$task/log.txt
done

# ########################################################################

echo "#################################################################"
echo "Final Parsing Model Integrated with Pretrained Encoders of Auxiliary tasks..."
echo "#################################################################"
touch saved_models/$model_path/log.txt
python examples/GraphParser.py --dataset ud --domain $domain --rnn_mode LSTM \
--num_epochs $num_epochs --batch_size 16 --hidden_size 512 \
--arc_space 512 --arc_tag_space 128 --num_layers 2 --num_filters 100 --use_char  \
--word_dim $word_dim --char_dim 100 --pos_dim 100 --initializer xavier --opt adam \
--learning_rate 0.002 --decay_rate 0.5 --schedule 6 --clip 5.0 --gamma 0.0 --epsilon 1e-6 \
--p_rnn 0.33 0.33 --p_in 0.33 --p_out 0.33 --arc_decode mst --pos_embedding random \
--punct_set '.' '``'  ':' ','  --word_embedding fasttext --char_embedding random --word_path $word_path  \
--gating --num_gates 4 \
--set_num_training_samples $set_num_training_samples \
--load_sequence_taggers_paths saved_models/$model_path/add_label/domain_$domain.pt \
saved_models/$model_path/Multitask_case_predict/domain_$domain.pt \
saved_models/$model_path/Multitask_POS_predict/domain_$domain.pt \
--model_path saved_models/$model_path/final_ensembled_BiAFF_LCM 2>&1 | tee saved_models/$model_path/log.txt
mv saved_models/$model_path/log.txt saved_models/$model_path/final_ensembled_BiAFF_LCM/log.txt

end_time=`date +%s`
echo execution time was `expr $end_time - $start_time` s.

