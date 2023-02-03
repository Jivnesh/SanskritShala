
domain="VST"
# Path of pretrained embedding file
word_path=data/cc.VST.300.txt
# Path to store model and predictions  
saved_models=./saved_models
declare -i num_epochs=100
declare -i word_dim=300
start_time=`date +%s`
current_time=$(date "+%Y.%m.%d-%H.%M.%S")
model_path="VST_"$current_time
touch $saved_models/base_log.txt

##############################################################
## Running the base Biaffine Parser
echo "#################################################################"
echo "Currently base model (Oracle MI) in progress..."
echo "#################################################################"
python examples/GraphParser_MTL_POS.py --dataset ud --domain $domain --rnn_mode LSTM \
--num_epochs $num_epochs  --batch_size 16 --hidden_size 512 --arc_space 512 \
--arc_tag_space 128 --num_layers 2 --num_filters 100 --use_char --use_pos \
--word_dim $word_dim --char_dim 100 --pos_dim 100 --initializer xavier --opt adam \
--learning_rate 0.002 --decay_rate 0.5 --schedule 6 --clip 5.0 --gamma 0.0 \
--epsilon 1e-6 --p_rnn 0.33 0.33 --p_in 0.33 --p_out 0.33 --arc_decode mst \
--punct_set '.' '``'  ':' ','  --word_embedding fasttext --char_embedding random  --pos_embedding random --word_path $word_path \
--model_path $saved_models/$model_path 2>&1 | tee $saved_models/base_log.txt
mv $saved_models/base_log.txt $saved_models/$model_path/base_log.txt

# ###################################################################
# Pretraining step: Running the Sequence Tagger
# Auxiliary tasks : 'Multitask_case_predict' 'Multitask_POS_predict' 'Multitask_label_predict'
for task in 'Multitask_case_predict' 'Multitask_POS_predict' 'Multitask_label_predict'; do
	touch $saved_models/$model_path/log.txt
	echo "#################################################################"
	echo "Currently $task in progress..."
	echo "#################################################################"  
    python examples/SequenceTagger.py --dataset ud --domain $domain --task $task \
    --rnn_mode LSTM --num_epochs $num_epochs --batch_size 16 --hidden_size 512 \
	--tag_space 128 --num_layers 2 --num_filters 100 --use_char --use_pos  --char_dim 100 \
	--pos_dim 100 --initializer xavier --opt adam --learning_rate 0.002 --decay_rate 0.5 \
	--schedule 6 --clip 5.0 --gamma 0.0 --epsilon 1e-6 --p_rnn 0.33 0.33 \
	--p_in 0.33 --p_out 0.33 --punct_set '.' '``'  ':' ','  \
	--use_unlabeled_data --use_labeled_data \
	--word_dim $word_dim --word_embedding fasttext --word_path $word_path --pos_embedding random \
	--parser_path $saved_models/$model_path/ \
	--char_embedding random \
	--model_path $saved_models/$model_path/$task/ 2>&1 | tee $saved_models/$model_path/log.txt
	mv $saved_models/$model_path/log.txt $saved_models/$model_path/$task/log.txt
done

#####################################################################
## Integration step: The final ensembled proposed system

echo "#################################################################"
echo "Currently final model in progress..."
echo "#################################################################"
# --load_path saved_models/$model_path/Multitask_POS_predict/domain_$domain.pt \
touch $saved_models/$model_path/log.txt
python examples/GraphParser_MTL_POS.py --dataset ud --domain $domain --rnn_mode LSTM \
--num_epochs $num_epochs --batch_size 16 --hidden_size 512 \
--arc_space 512 --arc_tag_space 128 --num_layers 2 --num_filters 100 --use_char --use_pos \
--word_dim $word_dim --char_dim 100 --pos_dim 100 --initializer xavier --opt adam \
--learning_rate 0.002 --decay_rate 0.5 --schedule 6 --clip 5.0 --gamma 0.0 --epsilon 1e-6 \
--p_rnn 0.33 0.33 --p_in 0.33 --p_out 0.33 --arc_decode mst --pos_embedding random \
--punct_set '.' '``'  ':' ','  --word_embedding fasttext --char_embedding random --word_path $word_path  \
--gating --num_gates 4 \
--load_path $saved_models/$model_path/Multitask_POS_predict/domain_$domain.pt \
--load_sequence_taggers_paths $saved_models/$model_path/Multitask_case_predict/domain_$domain.pt \
$saved_models/$model_path/Multitask_POS_predict/domain_$domain.pt \
$saved_models/$model_path/Multitask_label_predict/domain_$domain.pt \
--model_path $saved_models/$model_path/final_ensembled_TranSeq 2>&1 | tee $saved_models/$model_path/log.txt
mv $saved_models/$model_path/log.txt $saved_models/$model_path/final_ensembled_TranSeq/log.txt

python examples/VST_Pred_Prepare.py $model_path
python examples/VST_macro_score.py $model_path

end_time=`date +%s`
echo execution time was `expr $end_time - $start_time` s.