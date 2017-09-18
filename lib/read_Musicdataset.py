__author__ = 'charlie'
import glob
import os
import random

from six.moves import cPickle as pickle
from tensorflow.python.platform import gfile


def read_dataset(data_dir):
    pickle_filename = "music_data.pickle"
    pickle_filepath = os.path.join(data_dir, pickle_filename)
    #pickle_filepath = pickle_filename
    if not os.path.exists(pickle_filepath):
        #utils.maybe_download_and_extract(data_dir, DATA_URL, is_zipfile=True)
        result = create_image_lists(os.path.join(data_dir, "images"))
        print ("Pickling ...")
        with open(pickle_filepath, 'wb') as f:
            pickle.dump(result, f, pickle.HIGHEST_PROTOCOL)
    else:
        print ("Found pickle file!")

    with open(pickle_filepath, 'rb') as f:
        result = pickle.load(f)
        training_records = result['training']
        validation_records = result['validation']
        del result

    return training_records, validation_records


def create_image_lists(image_dir):
    if not gfile.Exists(image_dir):
        print("Image directory'" + image_dir + "' not found.")
        return None
    directories = ['training', 'validation']
    image_list = {}

    for directory in directories:
        file_list = []
        image_list[directory] = []
        file_glob = os.path.join(image_dir, directory, '*.' + 'png')
        file_list.extend(glob.glob(file_glob))

        if not file_list:
            print('No files found')
        else:
            for f in file_list:
                filename = os.path.splitext(f.split("/")[-1])[0]
                m_annotation_file = os.path.join(image_dir,"..", "m_annotations", directory, filename + '.png')
                o_annotation_file = os.path.join(image_dir, "..", "o_annotations", directory, filename + '.png')
                if os.path.exists(m_annotation_file) and os.path.exists(o_annotation_file):
                    record = {'image': f, 'm_annotation': m_annotation_file, 'o_annotation': o_annotation_file, 'filename': filename}
                    image_list[directory].append(record)
                else:
                    print("Annotation file not found for %s - Skipping" % filename)

        random.shuffle(image_list[directory])
        no_of_images = len(image_list[directory])
        print ('No. of %s files: %d' % (directory, no_of_images))

    return image_list
