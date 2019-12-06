import tensorflow as tf
import grpc
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc
from tensorflow.contrib import util as contrib_util
import numpy as np

label = {0: 'animal', 1: 'flower', 2: 'guitar', 3: 'houses', 4: 'plane'}


def preprocess_image(img, height, width, scope=None):
    with tf.name_scope(scope, 'inference_image', [img, height, width]):
        if img.dtype != tf.float32:
            img = tf.image.convert_image_dtype(img, dtype=tf.float32)
        if height and width:
            # Resize the image to the specified height and width.
            img = tf.expand_dims(img, 0)
            img = tf.image.resize_bilinear(img, [height, width], align_corners=False)  # 不对齐角落
            img = tf.squeeze(img, [0])
        img = tf.subtract(img, 0.5)
        img = tf.multiply(img, 2.0)
        return img


def main():
    with tf.Session() as sess:
        image_string = tf.gfile.FastGFile('./tmp/data/test_image/flower.jpg', 'rb').read()
        image = tf.image.decode_jpeg(image_string, channels=3)
        processed_image = preprocess_image(image, 299, 299)
        processed_image = tf.expand_dims(processed_image, 0)
        img = sess.run(processed_image)

    server = '0.0.0.0:9000'
    channel = grpc.insecure_channel(server)
    stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)
    request = predict_pb2.PredictRequest()
    request.model_spec.name = 'test'
    request.model_spec.signature_name = 'predict_images'
    request.inputs['input'].CopyFrom(
        contrib_util.make_tensor_proto(img, shape=[1, 299, 299, 3]))
    result = stub.Predict(request)
    probabilities = np.array(result.outputs['output'].float_val)
    sorted_inds = [i[0] for i in sorted(enumerate(-probabilities), key=lambda x: x[1])]
    for i in range(5):
        index = sorted_inds[i]
        print('Probability %0.2f%% => [%s]' % (probabilities[index] * 100, label[index]))


if __name__ == '__main__':
    main()
