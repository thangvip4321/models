# Instruction on training DeepLab with custom dataset ( specifically Supervisely)


### 1. Define these values:
- ${YOUR_DATASET_DIR} : directory to place your dataset
- ${TF_RECORD_DIR}: directory that will contain your dataset in TFRecord format
- ${INDEX_DIR} : index directory, to include what folder in your dataset will be converted to TFRecord format, more info below
- ${YOUR_LOG_DIR} : place to log your training
  
### 2. Create your dataset in this format:
```
${YOUR_DATASET_DIR}
  +train
    +img
    +masks_machine (these folders' name are hard-coded in `build_custom_dataset.py` , change it if you want)
  +${INDEX_DIR}
    + train.txt
    + test.txt ( these two below are optional, depends on  your dataset having test and val dir or not)
    + val.txt
  +${TF_RECORD_DIR}

```

### 3. 
`python3 datasets/build_custom_dataset.py --raw_dataset_dir ${YOUR_DATASET_DIR} --output_dir ${TF_RECORD_DIR} --index_dir ${INDEX_DIR}`

### 4. 
`python3 deeplab/train.py --train_logdir ${YOUR_LOG_DIR} --dataset custom_seg --dataset_dir ${TF_RECORD_DIR} --atrous-rate 2`
