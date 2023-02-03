

echo "Currently Intrinsic Evaluations in progress..."
cd evaluations/Intrinsic
for model in 'word2vec' 'glove' 'FastText' 'CharLM'  'ELMO' 'Albert' ; do
	python main.py  --model $model
done
