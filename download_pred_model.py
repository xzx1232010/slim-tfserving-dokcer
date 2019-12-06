from datasets import dataset_utils
import tensorflow as tf

url = "http://download.tensorflow.org/models/inception_v3_2016_08_28.tar.gz"
checkpoints_dir = './tmp/checkpoints'

if not tf.gfile.Exists(checkpoints_dir):
    tf.gfile.MakeDirs(checkpoints_dir)

dataset_utils.download_and_uncompress_tarball(url, checkpoints_dir)
