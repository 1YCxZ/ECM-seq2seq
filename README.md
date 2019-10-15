Emotional Chatting Machine
==========================
This reponsitory includes two models
- a basic seq2seq model with attention and beamsearch
- ECM model

Thanks to https://github.com/AaronYALai/Seq2seqAttn_ECM, I learn a lot from this reponsitory.
However, I can't get a satisfactory results on the chinese corpus when I use AaronYALai's model.
So, I implement a new ECM model based on the [tensorflow seq2seq API](https://tensorflow.google.cn/api_docs/python/tf/contrib/seq2seq).

## Requirements
- python 2.7
- tensorflow == 1.4

## Sample data
sample data here is only for showing the data format, not for training.
- category: target sentence emotion category
- choice: target sentence emotional word annotation
- source: source sentence
- target: target sentence

## Just tell me how it works

### Set up work space
Create a new folder by following the parameters "workspace" in the yaml configuration file.
for example:
```
./works/example/
```

### Prepare your data and configuration file
you can check the sample data folder for the data format.Then you need to put them under path like
```
./works/example/data/
```
### Train the model
for training the basic model:
```
python train.py
```
for training ECM model:
```
python train_ECM.py
```
### Infer
- "infer_ECM.py" will first create a calculation graph of infer model then load the trained parameters, and finally perform the inference, which is not suitable for deployment.
```
python infer_ECM.py
```
- "save_infer_model.py" will first create a calculation graph, load the training parameters, and then save the infer model as a model file.
At this time, you can use different languages of tensorflow API(C++/Java) to load the infer model.
```
python save_infer_model.py
python infer_ECM_deploy.py
```

## Model performance
The following result is based on the dataset I crawled from BaiDu Tieba, including 3 emotion types:
- no emotion: 321052
- pos emotion: 137086
- neg emotion: 240233

##### Parameters:
  ```
  embeddings:
     embed_size: 300
     vocab_size: 40000
  encoder:
     bidirectional: True
     cell_type: LSTM
     num_layers: 2
     num_units: 512
  decoder:
     attn_num_units: 512
     cell_type: LSTM
     num_layers: 2
     num_units: 512
     state_pass: True
     infer_max_iter: 25
     emo_cat_emb_size: 256
     emo_internal_memory_units: 256
     num_emotion: 3
  ```
##### Training perplexity
![Image text](./training_perplexity.png)

## Extra dataset
NTCIR
Short Text Conversation Task(STC-3)
chinese Emotional Conversation Generation (CECG) Subtask
http://coai.cs.tsinghua.edu.cn/hml/challenge/dataset_description/
You can also find the dataset in this project.
