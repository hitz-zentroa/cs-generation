python fine-tuning/train_decoder.py --dataset_path $1 --save_path $2 --model "Undi95/Meta-Llama-3-8B-hf" --model_type "llama" --lr 0.0005
python fine-tuning/train_decoder.py --dataset_path $1 --save_path $2 --model "mistralai/Mistral-7B-v0.3" --model_type "mistral" --lr 0.0005
python fine-tuning/train_instruct.py --dataset_path $1 --save_path $2 --model "Undi95/Meta-Llama-3-8B-Instruct-hf" --model_type "llama" --lr 0.0005
python fine-tuning/train_instruct.py --dataset_path $1 --save_path $2 --model "mistralai/Mistral-7B-Instruct-v0.3" --model_type "mistral" --lr 0.0005