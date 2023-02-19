
domain="san"
word_path="../Documents/DCST/data/multilingual_word_embeddings/cc.sanskrit.300.new.vec"
char_path="../Documents/DCST/data/multilingual_word_embeddings/hellwig_char_embedding.128"
pos_path='../Documents/DCST/data/multilingual_word_embeddings/pos_embedding_FT.100'
declare -i num_epochs=1
declare -i word_dim=300
declare -i set_num_training_samples=500
start_time=`date +%s`
current_time=$(date "+%Y.%m.%d-%H.%M.%S")
# model_path="ud_parser_san_"$current_time 

# ######## Changes ###########
# ## --use_pos is removed from all the models
# ## singletions are excluded.
# ##  auxiliary task epoch change to 20

touch saved_models/log.txt
################################################################ 
echo "#################################################################"
echo "Currently base model in progress..."
echo "#################################################################"
python examples/GraphParser.py --dataset ud --domain $domain --rnn_mode LSTM \
--num_epochs $num_epochs --batch_size 16 --hidden_size 512 --arc_space 512 \
--arc_tag_space 128 --num_layers 3 --num_filters 100 --use_char  \
--word_dim $word_dim --pos_dim 100 --initializer xavier --opt adam \
--learning_rate 0.002 --decay_rate 0.5 --schedule 6 --clip 5.0 --gamma 0.0 \
--epsilon 1e-6 --p_rnn 0.33 0.33 --p_in 0.33 --p_out 0.33 --arc_decode mst \
--punct_set '.' '``'  ':' ','  --word_embedding fasttext  \
--set_num_training_samples $set_num_training_samples \
--word_path $word_path --char_path $char_path --pos_path $pos_path  --pos_embedding random --char_embedding random --char_dim 100 \
--model_path saved_models/$model_path 2>&1 | tee saved_models/log.txt
mv saved_models/log.txt saved_models/$model_path/log.txt


## model_path="ud_parser_san_2020.03.03-12.24.41"

####### self-training ###########
## 'number_of_children' 'relative_pos_based' 'distance_from_the_root'

####### Multitask setup #########
## 'Multitask_label_predict' 'Multitask_case_predict' 'Multitask_POS_predict' 'Multitask_coarse_predict' 
## 'predict_ma_tag_of_modifier' 'add_label' 'predict_case_of_modifier' 

####### Other Tasks #############
## #   'add_head_ma' 'add_head_coarse_pos' 'predict_coarse_of_modifier' 

for task in 'Multitask_label_predict' 'Multitask_case_predict' 'Multitask_POS_predict' 'number_of_children' 'relative_pos_based' 'distance_from_the_root' ; do
	echo "#################################################################"
	echo "Currently $task in progress..."
	echo "#################################################################"  
	touch saved_models/log.txt
    python examples/SequenceTagger.py --dataset ud --domain $domain --task $task \
    --rnn_mode LSTM --num_epochs $num_epochs --batch_size 16 --hidden_size 512 \
	--tag_space 128 --num_layers 3 --num_filters 100 --use_char  \
	--pos_dim 100 --initializer xavier --opt adam --learning_rate 0.002 --decay_rate 0.5 \
	--schedule 6 --clip 5.0 --gamma 0.0 --epsilon 1e-6 --p_rnn 0.33 0.33 \
	--p_in 0.33 --p_out 0.33 --punct_set '.' '``'  ':' ','  \
	--word_dim $word_dim --word_embedding fasttext --word_path $word_path \
	--parser_path saved_models/$model_path/ \
	--use_unlabeled_data --char_path $char_path --pos_path $pos_path --pos_embedding random --char_embedding random --char_dim 100  \
	--model_path saved_models/$model_path/$task/ 2>&1 | tee saved_models/log.txt
	mv saved_models/log.txt saved_models/$model_path/$task/log.txt
done

echo "#################################################################"
echo "Currently final model in progress..."
echo "#################################################################"
touch saved_models/log.txt
python examples/GraphParser.py --dataset ud --domain $domain --rnn_mode LSTM \
--num_epochs $num_epochs --batch_size 16 --hidden_size 512 \
--arc_space 512 --arc_tag_space 128 --num_layers 3 --num_filters 100 --use_char  \
--word_dim $word_dim --pos_dim 100 --initializer xavier --opt adam \
--learning_rate 0.002 --decay_rate 0.5 --schedule 6 --clip 5.0 --gamma 0.0 --epsilon 1e-6 \
--p_rnn 0.33 0.33 --p_in 0.33 --p_out 0.33 --arc_decode mst \
--punct_set '.' '``'  ':' ','  --word_embedding fasttext \
--word_path $word_path --gating --num_gates 4 \
--char_path $char_path --pos_path $pos_path  --pos_embedding random --char_embedding random --char_dim 100 \
--load_sequence_taggers_paths saved_models/$model_path/Multitask_case_predict/domain_$domain.pt \
saved_models/$model_path/Multitask_POS_predict/domain_$domain.pt \
saved_models/$model_path/Multitask_label_predict/domain_$domain.pt \
--model_path saved_models/$model_path/final_ensembled_multi/ 2>&1 | tee saved_models/log.txt
mv saved_models/log.txt saved_models/$model_path/final_ensembled_multi/log.txt
end_time=`date +%s`
echo execution time was `expr $end_time - $start_time` s.


echo "#################################################################"
echo "Currently final model in progress..."
echo "#################################################################"
touch saved_models/log.txt
python examples/GraphParser.py --dataset ud --domain $domain --rnn_mode LSTM \
--num_epochs $num_epochs --batch_size 16 --hidden_size 512 \
--arc_space 512 --arc_tag_space 128 --num_layers 3 --num_filters 100 --use_char  \
--word_dim $word_dim --pos_dim 100 --initializer xavier --opt adam \
--learning_rate 0.002 --decay_rate 0.5 --schedule 6 --clip 5.0 --gamma 0.0 --epsilon 1e-6 \
--p_rnn 0.33 0.33 --p_in 0.33 --p_out 0.33 --arc_decode mst \
--punct_set '.' '``'  ':' ','  --word_embedding fasttext \
--word_path $word_path --gating --num_gates 7 \
--char_path $char_path --pos_path $pos_path  --pos_embedding random --char_embedding random --char_dim 100 \
--load_sequence_taggers_paths saved_models/$model_path/Multitask_case_predict/domain_$domain.pt \
saved_models/$model_path/Multitask_POS_predict/domain_$domain.pt \
saved_models/$model_path/Multitask_label_predict/domain_$domain.pt \
saved_models/$model_path/number_of_children/domain_$domain.pt \
saved_models/$model_path/relative_pos_based/domain_$domain.pt \
saved_models/$model_path/distance_from_the_root/domain_$domain.pt \
--model_path saved_models/$model_path/final_ensembled_multi_self/ 2>&1 | tee saved_models/log.txt
mv saved_models/log.txt saved_models/$model_path/final_ensembled_multi_self/log.txt
end_time=`date +%s`
echo execution time was `expr $end_time - $start_time` s.