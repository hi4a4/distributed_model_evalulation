import sys
from os import walk

import pandas as pd

from config import config
from predict.evaluate_prediction_vid import evaluate

'''
Modify sql or aws code from predict.py and evaluate_prediction_vid.py
Update/Insert statement - remove
select statements - use read_csv now
'''


def get_model_data(model_name):
    data = pd.read_csv(config.DATABASE_FOLDER + '/models.csv')
    return data[data.name == model_name]


def get_model_collections(collectionid_list):
    data = pd.read_csv(config.DATABASE_FOLDER + '/concept_intermediate.csv')
    # key: -colection id values: concepts ids in the collection
    collection_conceptids_list = {}
    for collectionid in collectionid_list:
        conceptids = data[data.id == collectionid].conceptid.values
        collection_conceptids_list[-collectionid] = conceptids
    return collection_conceptids_list


def cast_to_list(string_list):
    return [int(word) for word in string_list.replace('}', '').replace('{', '').split(',')]


def get_verification_videos(ids):
    data = pd.read_csv(config.DATABASE_FOLDER + '/videos.csv')
    filenames = []
    for id in ids:
        filenames.append(data[data.id == id].filename.values[0])
    return filenames


def main():
    if len(sys.argv) != 5:
        print('After script name specify: video filename, model weights filename, start frame, and end frame')
        return

    video_filename = sys.argv[1].split('/')[-1]
    model_filename = sys.argv[2].split('/')[-1]
    start_frame = int(sys.argv[3])
    end_frame = int(sys.argv[4])

    model_name = '-'.join(model_filename.split('-')[:-1])
    model_data = get_model_data(model_name)
    concepts = cast_to_list(model_data['concepts'].iloc[0])
    concept_collections = cast_to_list(model_data['concept_collections'].iloc[0])
    collections = get_model_collections(concept_collections)

    if 2136 not in concepts or 2124 not in concepts:
        print("Model Has not collection, so skipping...")
        return

    evaluate(video_filename, model_filename, concepts, collections, start_frame, end_frame)


main()
