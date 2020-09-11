import tensorflow as tf 
import os
import math as m
from datasets import build_data
import sys
flags = tf.app.flags 
FLAGS = flags.FLAGS

flags.DEFINE_string('raw_dataset_dir',"../my_dataset","Folder of your dataset, which include `train`,`test`,`val` subfolders")
flags.DEFINE_string('output_dir',"../output","Directory for preprocessed dataset")
flags.DEFINE_string('index_dir','./text.txt','Directory containing image directory name, e.g: `train` ,`val`...')
NUM_SHARDS = 3

def convert_data(index):
    #extract raw data to tfrecord format from the folder named after `index`
    dataset_index = os.path.basename(index)[:-4]
    print("wow",dataset_index)
    image_dir_path = os.path.join(FLAGS.raw_dataset_dir,dataset_index,'img')
    #label_dir_path = os.path.join(FLAGS.raw_dataset_dir,dataset_index,'label')
    if not os.path.exists(image_dir_path):
        return
    print("path",image_dir_path)#,label_dir_path)
    image_list = list(map(lambda filename : os.path.basename(filename).split(sep='.')[0] ,tf.io.gfile.listdir(image_dir_path)))
    num_images = len(image_list)
    num_per_shard = m.ceil(num_images/NUM_SHARDS)
    image_reader = build_data.ImageReader('jpg', channels=3)
    label_reader = build_data.ImageReader('png', channels=1)
    for shard_id in range(NUM_SHARDS):
        output_filename = os.path.join(
            FLAGS.output_dir,dataset_index,
            '%s-%05d-of-%05d.tfrecord' % (dataset_index, shard_id, NUM_SHARDS))
        with tf.io.TFRecordWriter(output_filename) as tfrecord_writer:
            start_idx = shard_id * num_per_shard
            end_idx = min((shard_id + 1) * num_per_shard, num_images)
            for i in range(start_idx, end_idx):
                sys.stdout.write('\r>> Converting image %d/%d shard %d' % (
                    i + 1, num_images, shard_id))
                sys.stdout.flush()
                # Read the image.
                image_filename = os.path.join(
                    FLAGS.raw_dataset_dir,dataset_index,"img", image_list[i]+"."+FLAGS.image_format)
                image_data = tf.io.gfile.GFile(image_filename, 'rb').read()
                height, width = image_reader.read_image_dims(image_data)
                # Read the semantic segmentation annotation.
                seg_filename = os.path.join(
                    FLAGS.raw_dataset_dir,dataset_index,"masks_machine",
                    image_list[i] + '.' + FLAGS.label_format)
                seg_data = tf.gfile.GFile(seg_filename, 'rb').read()
                seg_height, seg_width = label_reader.read_image_dims(seg_data)
                if height != seg_height or width != seg_width:
                    raise RuntimeError('Shape mismatched between image and label.')
                # Convert to tf example.
                example = build_data.image_seg_to_tfexample(
                    image_data, image_list[i], height, width, seg_data)
                tfrecord_writer.write(example.SerializeToString())
        sys.stdout.write('\n')
        sys.stdout.flush()



def main(unused_arg):
    print(FLAGS)
    index_names = tf.io.gfile.glob(os.path.join(FLAGS.index_dir,"*.txt"))
    print(index_names)
    for index_name in index_names:
        convert_data(index_name)

if __name__ == '__main__':
    tf.app.run()
