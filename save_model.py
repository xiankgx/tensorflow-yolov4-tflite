import tensorflow as tf
from absl import app, flags, logging
from absl.flags import FLAGS

import core.utils as utils
from core.config import cfg
from core.yolov4 import YOLO, decode, filter_boxes

flags.DEFINE_string('weights',
                    './scripts/yolov4.weights',
                    'path to weights file')
flags.DEFINE_string('output', './saved_models/yolov4', 'path to output')
flags.DEFINE_boolean('tiny', False, 'is yolo-tiny or not')
flags.DEFINE_integer('input_size', 512, 'define input size of export model')
flags.DEFINE_float('score_thres', 0.2, 'define score threshold')
flags.DEFINE_string('framework',
                    'tf',
                    'define what framework do you want to convert (tf, trt, tflite)')
flags.DEFINE_string('model', 'yolov4', 'yolov3 or yolov4')


def save_tf():
    STRIDES, ANCHORS, NUM_CLASS, XYSCALE = utils.load_config(FLAGS)

    input_layer = tf.keras.layers.Input(
        [FLAGS.input_size, FLAGS.input_size, 3]
    )
    feature_maps = YOLO(input_layer, NUM_CLASS, FLAGS.model, FLAGS.tiny)

    bbox_tensors = []
    prob_tensors = []
    for i, fm in enumerate(feature_maps):
        output_tensors = decode(fm,
                                FLAGS.input_size
                                // (2 * ((4 if FLAGS.tiny else 3) + i)),
                                NUM_CLASS,
                                STRIDES,
                                ANCHORS,
                                i,
                                XYSCALE,
                                FLAGS.framework)
        bbox_tensors.append(output_tensors[0])
        prob_tensors.append(output_tensors[1])
    pred_bbox = tf.concat(bbox_tensors, axis=1)
    pred_prob = tf.concat(prob_tensors, axis=1)

    if FLAGS.framework == 'tflite':
        pred = (pred_bbox, pred_prob)
    else:
        boxes, pred_conf = filter_boxes(pred_bbox, pred_prob,
                                        score_threshold=FLAGS.score_thres,
                                        input_shape=tf.constant([FLAGS.input_size, FLAGS.input_size]))
        pred = tf.concat([boxes, pred_conf], axis=-1)
    model = tf.keras.Model(input_layer, pred)

    print(f"Restoring model weights from: {FLAGS.weights}")
    utils.load_weights(model, FLAGS.weights, FLAGS.model, FLAGS.tiny)

    # Restore model weights from TF training checkpoints
    checkpoint_kwargs = {
        "model": model,
    }
    checkpoint = tf.train.Checkpoint(**checkpoint_kwargs)
    ckpt_manager = tf.train.CheckpointManager(checkpoint,
                                              "checkpoints",
                                              max_to_keep=5)
    # if a checkpoint exists, restore the latest checkpoint.
    if ckpt_manager.latest_checkpoint:
        checkpoint.restore(ckpt_manager.latest_checkpoint)
        print("Latest checkpoint restored!")
    else:
        print("Initializing from scratch.")

    model.summary()
    model.save(FLAGS.output)


def main(_argv):
    save_tf()


if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
