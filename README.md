# slim-tfserving-docker

### 环境

Python3、tensorflow-1.14.0、ubuntu16.04

### 1.数据集转换tfrecoed格式

数据集目录：./tmp/data

命令：`python image_to_tf.py`，在./tmp/data/tfrecord目录下，会生成相对应的tfrecord文件和标签文件label.txt。

### 2.下载inception_v3预训练模型

命令：`python download_pred_model.py`，在./tmp/checkpoints目录下，会有相应的inception_v3.ckpt文件，如果要下载其他预训练模型，只要更改代码里相应的url即可。

### 3.训练模型

命令：`python train_image_classifier.py \   
   			--train_dir=./tmp/train_logs \ 
    		--dataset_name=classify \
    		--dataset_split_name=train \
    		--dataset_dir=./tmp/data/tfrecord \
    		--model_name=inception_v3 \
    		--checkpoint_path=./tmp/checkpoints/inception_v3.ckpt \
    		--checkpoint_exclude_scopes=InceptionV3/Logits,InceptionV3/AuxLogits \
    		--trainable_scopes=InceptionV3/Logits,InceptionV3/AuxLogits \
    		--batch_size=4 \
    		--clone_on_cpu=True \
    		--optimizer=adam \
    		--max_number_of_steps=100`

|           参数            |                        说明                        |
| :-----------------------: | :------------------------------------------------: |
|         train_dir         |                    保存模型路径                    |
|       dataset_name        |                      数据集名                      |
|    dataset_split_name     |                   训练集、测试集                   |
|        dataset_dir        |                     数据集路径                     |
|        model_name         |                       模型名                       |
|      checkpoint_path      |                   预训练模型路径                   |
| checkpoint_exclude_scopes |                   去除哪些网络层                   |
|     trainable_scopes      |          只更新哪些参数，不填更新所有参数          |
|     log_every_n_steps     |              每隔多少步打印一次loss值              |
|       clone_on_cpu        |                    是否使用cpu                     |
|    save_summaries_secs    | 每隔多少秒记录信息到summaries，可用tensorboard查看 |
|    max_number_of_steps    |                      训练步数                      |
|    save_interval_secs     |               每隔多少秒保存模型参数               |

### 4.评价模型

命令：`python eval_image_classifier.py \
    		--alsologtostderr \
    		--checkpoint_path=./tmp/train_logs/model.ckpt-100 \
   			--dataset_dir=./tmp/data/tfrecord \
    		--dataset_name=classify \
    		--dataset_split_name=validation \
    		--model_name=inception_v3`

### 5.tensorboard查看

命令：`tensorboard --logdir=./tmp/train_logs`

### 6.导出模型

先加载网络结构，导出pb模型，再冻结变量，将变量和网络结构整合导出到pb模型里。

命令：`python export_inference_graph.py \
  			--alsologtostderr \
  			--model_name=inception_v3 \
 	 		--output_file=./tmp/inception_v3_inf_graph.pb \
  			--dataset_name=classify \
  			--dataset_dir=./tmp/data/tfrecord`

命令：`python freeze_graph.py \
  			--input_graph=./tmp/inception_v3_inf_graph.pb \
  			--input_checkpoint=./tmp/train_logs/model.ckpt-100 \
  			--input_binary=true \
  			--output_graph=./tmp/frozen_inception_v3.pb \
  			--output_node_names=InceptionV3/Predictions/Softmax`

|       参数        |            说明             |
| :---------------: | :-------------------------: |
|    input_graph    |        网络结构模型         |
| input_checkpoint  |       训练的ckpt模型        |
|   input_binary    | input_graph是否为二进制模型 |
|   output_graph    |      输出冻结模型路径       |
| output_node_names |        保存节点名称         |

### 7.查看网络结构节点名称、单张图片预测

命令：`python inference.py`

### 8.模型转化savemodel格式

命令：`python pb_to_tfserving.py`

### 9.运行tensorflow serving容器

下拉镜像：`docker pull tensorflow/serving:1.14.0`

运行容器：`docker run -p 9000:8500  \
  --mount type=bind,source=/your_path/slim/tmp/saved_model_builder,target=/models/test \
  -e MODEL_NAME=test -t tensorflow/serving:1.14.0 &`

### 10.grpc远程调用模型

命令：`python grpc_tfserving.py`