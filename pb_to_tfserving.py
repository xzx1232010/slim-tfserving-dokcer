import tensorflow as tf


def save_model_predict():
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

    with tf.Session() as sess:
        image_string = tf.gfile.FastGFile('./tmp/data/test_image/flower.jpg', 'rb').read()
        image = tf.image.decode_jpeg(image_string, channels=3)
        processed_image = preprocess_image(image, 299, 299)
        processed_images = tf.expand_dims(processed_image, 0)
        img = sess.run(processed_images)
        sess.close()

    with tf.Session() as sess:
        model_path = './tmp/saved_model_builder/1/'
        meta_graph_def = tf.saved_model.loader.load(sess, [tf.saved_model.SERVING], model_path)
        # 获得签名
        signature = meta_graph_def.signature_def
        # 从签名获得张量名字
        in_tensor_name = signature['predict_images'].inputs['input'].name
        out_tensor_name = signature['predict_images'].outputs['output'].name
        # 获得张量
        in_tensor = sess.graph.get_tensor_by_name(in_tensor_name)
        out_tensor = sess.graph.get_tensor_by_name(out_tensor_name)
        # run
        print(sess.run(out_tensor, feed_dict={in_tensor: img}))
        print(in_tensor_name, out_tensor_name)


def change_model(model_path):
    # 加载模型
    with tf.gfile.FastGFile(model_path, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        tf.import_graph_def(graph_def, name='')

    with tf.Session() as sess:
        input_image_tensor = sess.graph.get_tensor_by_name("input:0")
        output_tensor_name = sess.graph.get_tensor_by_name('InceptionV3/Predictions/Softmax:0')

        output_path = './tmp/saved_model_builder/1/'
        builder = tf.saved_model.builder.SavedModelBuilder(output_path)
        inputs = {'input': tf.saved_model.utils.build_tensor_info(input_image_tensor)}
        outputs = {'output': tf.saved_model.utils.build_tensor_info(output_tensor_name)}

        method_name = tf.saved_model.PREDICT_METHOD_NAME
        prediction_signature = tf.compat.v1.saved_model.signature_def_utils.build_signature_def(inputs, outputs,
                                                                                                method_name)
        builder.add_meta_graph_and_variables(
            sess, [tf.saved_model.SERVING],
            signature_def_map={'predict_images': prediction_signature})
        builder.save()
        sess.close()


def main():
    model_path = './tmp/frozen_inception_v3.pb'
    change_model(model_path)  # 模型转化save_model格式
    save_model_predict()  # 加载save_model格式预测图片


if __name__ == '__main__':
    main()
