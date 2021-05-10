# Passage Retrieval for Outside-Knowledge Visual Question Answering

This repository contains code and data for our paper [Passage Retrieval for Outside-Knowledge Visual Question Answering]()

## Data and checkpoints

Our data is based on the [OK-VQA](https://okvqa.allenai.org/index.html) dataset.
* First download all [OK-VQA](https://okvqa.allenai.org/index.html) files
* Then download the [collecton](https://ciir.cs.umass.edu/downloads/ORConvQA/all_blocks.txt.gz) file (all_blocks.txt)
* Finally, download other files [here](https://ciir.cs.umass.edu/downloads/okvqa/). 
    * passage_id_to_line_id.json: map passages ids to line ids in all_blocks.txt
    * data: train/val/test split and a small validation collection
    * okvqa.datasets: pre-extracted image features with [this script](https://github.com/huggingface/transformers/blob/master/examples/research_projects/lxmert/extracting_data.py)
    * (Optional) checkpoint: our model checkpoint. No need to download if you want to train your own model

## Sample commands

### Training, and evaluating on the validation set with the small validation collection
```
python -u -m torch.distributed.launch --nproc_per_node 4 train_retriever.py \
    --input_file=DATA_DIR/data/{}_pairs_cap_combine_sum.txt \
    --image_features_path=DATA_DIR/okvqa.datasets \
    --output_dir=OUTPUT_DIR \
    --ann_file=DATA_DIR/mscoco_val2014_annotations.json \
    --ques_file=DATA_DIR/OpenEnded_mscoco_val2014_questions.json \
    --passage_id_to_line_id_file=DATA_DIR/passage_id_to_line_id.json \
    --all_blocks_file=DATA_DIR/all_blocks.txt \
    --gen_passage_rep=False \
    --gen_passage_rep_input=DATA_DIR/val2014_blocks_cap_combine_sum.txt \
    --cache_dir=HUGGINGFACE_CACHE_DIR (optional) \
    --gen_passage_rep_output="" \
    --retrieve_checkpoint="" \
    --collection_reps_path="" \
    --val_data_sub_type=val2014 \
    --do_train=True \
    --do_eval=True \
    --do_eval_pairs=True \
    --per_gpu_train_batch_size=4 \
    --per_gpu_eval_batch_size=10 \
    --learning_rate=1e-5 \
    --num_train_epochs=2.0 \
    --logging_steps=5 \
    --save_steps=5000 \
    --eval_all_checkpoints=True \
    --overwrite_output_dir=False \
    --fp16=True \
    --load_small=False \
    --num_workers=4 \
    --query_encoder_type=lxmert \
    --query_model_name_or_path="unc-nlp/lxmert-base-uncased" \
    --lxmert_rep_type="{\"pooled_output\":\"none\"}" \
    --proj_size=768 \
    --neg_type=other_pos+all_neg
```

If experiencing OOM during evaluation, the evaluation process can be restarted with the same command but set `--do_train=False` and `--overwrite_output_dir=True`.

### Generating representations for all passages in the collection
```
python -u train_retriever.py \
    --input_file=DATA_DIR/data/{}_pairs_cap_combine_sum.txt \
    --image_features_path=DATA_DIR/okvqa.datasets \
    --output_dir=OUTPUT_DIR \
    --ann_file=DATA_DIR/mscoco_val2014_annotations.json \
    --ques_file=DATA_DIR/OpenEnded_mscoco_val2014_questions.json \
    --passage_id_to_line_id_file=DATA_DIR/passage_id_to_line_id.json \
    --all_blocks_file=DATA_DIR/all_blocks.txt \
    --gen_passage_rep=True \
    --gen_passage_rep_input=DATA_DIR/all_blocks.txt (or a split of this file) \
    --cache_dir=HUGGINGFACE_CACHE_DIR (optional) \
    --gen_passage_rep_output=OUTPUT_DIR_OF_PASSAGE_REPS \
    --retrieve_checkpoint=DIR_TO_YOUR_BEST_CHECKPOINT \
    --collection_reps_path="" \
    --val_data_sub_type=test2014 (doesn't matter here) \
    --do_train=False \
    --do_eval=False \
    --do_eval_pairs=False \
    --per_gpu_train_batch_size=4 \
    --per_gpu_eval_batch_size=300 \
    --learning_rate=1e-5 \
    --num_train_epochs=2.0 \
    --logging_steps=5 \
    --save_steps=5000 \
    --eval_all_checkpoints=True \
    --overwrite_output_dir=True \
    --fp16=True \
    --load_small=False \
    --num_workers=8 \
    --query_encoder_type=lxmert \
    --query_model_name_or_path=unc-nlp/lxmert-base-uncased \
    --lxmert_rep_type="{\"pooled_output\":\"none\"}" \
    --proj_size=768 \
    --neg_type=other_pos+all_neg
```

### Evaluating on the test set with the whole passage collection
```
python -u train_retriever.py \
    --input_file=DATA_DIR/data/{}_pairs_cap_combine_sum.txt \
    --image_features_path=DATA_DIR/okvqa.datasets \
    --output_dir=OUTPUT_DIR \
    --ann_file=DATA_DIR/mscoco_val2014_annotations.json \
    --ques_file=DATA_DIR/OpenEnded_mscoco_val2014_questions.json \
    --passage_id_to_line_id_file=DATA_DIR/passage_id_to_line_id.json \
    --all_blocks_file=DATA_DIR/all_blocks.txt \
    --gen_passage_rep=False \
    --gen_passage_rep_input=DATA_DIR/val2014_blocks_cap_combine_sum.txt \
    --cache_dir=HUGGINGFACE_CACHE_DIR (optional) \
    --gen_passage_rep_output="" \
    --retrieve_checkpoint=DIR_TO_YOUR_BEST_CHECKPOINT \
    --collection_reps_path=DIR_OF_PASSAGE_REPS \
    --val_data_sub_type=test2014 \
    --do_train=False \
    --do_eval=True \
    --do_eval_pairs=False \
    --per_gpu_train_batch_size=4 \
    --per_gpu_eval_batch_size=10 \
    --learning_rate=1e-5 \
    --num_train_epochs=2.0 \
    --logging_steps=5 \
    --save_steps=5000 \
    --eval_all_checkpoints=True \
    --overwrite_output_dir=True \
    --fp16=True \
    --load_small=False \
    --num_workers=1 \
    --query_encoder_type=lxmert \
    --query_model_name_or_path="unc-nlp/lxmert-base-uncased" \
    --lxmert_rep_type="{\"pooled_output\":\"none\"}" \
    --proj_size=768 \
    --neg_type=other_pos+all_neg
```


## Environment
Versions info available in `env.yml`

```
conda create --name okvqa python=3.8
conda install faiss-gpu cudatoolkit=10.1 -c pytorch
conda install pytorch torchvision torchaudio cudatoolkit=10.1 -c pytorch

git clone https://github.com/NVIDIA/apex
cd apex
export TORCH_CUDA_ARCH_LIST="5.2;6.1;7.5" (optional)
pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./

git clone https://github.com/huggingface/transformers.git
cd transformers
pip install -e .

conda install -c conda-forge tensorboard
pip install datasets
pip install pytrec_eval
conda install scikit-image
```

To generate image features, we need `opencv`, which doesn't work with python 3.8 at the moment. So image features are generated with an environment with python 3.7 (no need to do this if you have downloaded the features we extracted linked above). Use command `conda install -c conda-forge opencv` to install `opencv`.

## Acknowledgement
* Our training data is based on [OK-VQA](https://okvqa.allenai.org/index.html). We thank the OK-VQA authors for creating and releasing this useful resource.  
* `vqa_tools.py` is built on [VQA](https://github.com/GT-Vision-Lab/VQA). We thank the VQA authors for releasing their code.  
* `coco_tools.py` is built on [cocoapi](https://github.com/cocodataset/cocoapi). We thank the cocoapi authors for releasing their code.  

See copyright information in LICENSE.

## Citation
@inproceedings{prokvqa,
  title={{Passage Retrieval for Outside-Knowledge Visual Question Answering}},
  author={Chen Qu and and Hamed Zamani and Liu Yang and W. Bruce Croft and Erik Learned-Miller},
  booktitle={SIGIR},
  year={2021}
}