import os
import json
import uuid
import datetime
import copy
import time
import psutil

import pandas as pd
import numpy as np
from cv2 import cv2
from ffmpy import FFmpeg

from predict import predict
from config import config

'''
video_id - change to video_filename
model_username - same as weights file
concepts - need to get inside models.csv
upload_annotations - remove
user_id - remove
create_collection - remove
collections - get from get_conceptid_collections from concept_intermediate.csv
gpu_id - default to 0

Things I still need to do
Read in argv for video_filename and model_filename
Get concept from model_filename
Get collections
Test everything
'''


def get_videoid(filename):
    data = pd.read_csv(config.DATABASE_FOLDER + '/videos.csv')
    return data[data.filename == filename].id.values[0]


def evaluate(video_filename, model_filename, concepts, collections=None, start_frame=None, end_frame=None):
    # filename format: (video_id)_(model_name)-(version).mp4
    # This the generated video's filename
    filename = str(video_filename) + "_" + model_filename + ".mp4"

    video_id = get_videoid(video_filename)

    print("ai video id: filename: {0}: {1}".format(video_id, filename))

    print("Loading Video.")
    video_capture = get_video_capture(video_filename)
    tracking_id = 32

    results = predict.predict_on_video(model_filename, concepts, filename,
                                       video_capture, collections, start_frame, end_frame)

    if (results.empty):  # If the model predicts nothing stop here
        return
    print("done predicting")

    # Get human annotations
    validation = get_validation_set(video_capture.get(cv2.CAP_PROP_FPS), video_id,
                                    concepts, tracking_id, start_frame, end_frame)
    print(f'Got validation set shape: {validation.shape}')

    # This scores our well our model preformed against user annotations
    metrics, human_annotations = score_predictions(
        validation, results, config.EVALUATION_IOU_THRESH, concepts, collections)
    print('Got metrics')
    print(metrics)

    # Upload metrics
    model_name = '.'.join(model_filename.split('.')[:-1])
    save_metrics(metrics, video_id, model_name, start_frame, end_frame)
    print('Uploaded Metrics')
    # Generate video
    '''
    printing_with_time("Generating Video")
    generate_video(
        filename, video_capture,
        results, list(concepts) + list(collections.keys()), video_id, human_annotations)
    printing_with_time("Done generating")
    '''


def get_classmap(concepts):
    classmap = []
    data = pd.read_csv(config.DATABASE_FOLDER + '/concepts.csv')
    for concept_id in concepts:
        name = data[data.id == concept_id].iloc[0]['name']
        classmap.append([name, concepts.index(concept_id)])
    classmap = pd.DataFrame(classmap)
    classmap = classmap.to_dict()[0]
    return classmap


def printing_with_time(text):
    print(text + " " + str(datetime.datetime.now()))


def vectorized_iou(list_bboxes1, list_bboxes2):
    x11, y11, x12, y12 = np.split(list_bboxes1, 4, axis=1)
    x21, y21, x22, y22 = np.split(list_bboxes2, 4, axis=1)

    xA = np.maximum(x11, x21)
    yA = np.maximum(y11, y21)
    xB = np.minimum(x12, x22)
    yB = np.minimum(y12, y22)
    interArea = np.maximum((xB - xA), 0) * np.maximum((yB - yA), 0)
    boxAArea = np.abs((x12 - x11) * (y12 - y11))
    boxBArea = np.abs((x22 - x21) * (y22 - y21))
    denominator = (boxAArea + boxBArea - interArea)
    ious = np.where(denominator != 0, interArea / denominator, 0)
    return [iou[0] for iou in ious]


# These are the new functions for FP
def convert_hierarchy_fp_counts(value_counts, collections):
    # normal counts is a count_values type object
    # It ignores hierarchy counts
    normal_counts = copy.deepcopy(value_counts)
    for collectionid, count in value_counts[value_counts.index < 0].iteritems():
        del value_counts[collectionid]
        del normal_counts[collectionid]
        collection_conceptids = collections[collectionid]
        for conceptid in collection_conceptids:
            value_counts[conceptid] += count / len(collection_conceptids)
    return value_counts, normal_counts


# New function for convering hierarchy TP
def convert_hierarchy_tp_counts(pred_val_label_counts, HFP, collections, concepts):
    TP = dict(zip(concepts, [0.0] * len(concepts)))
    HTP = dict(zip(concepts, [0.0] * len(concepts)))
    for _, row in pred_val_label_counts.reset_index().iterrows():
        user_label = row.label_val  # Adding TP to this
        # If this is negative, add to TP 1/len(collection) and add to FP 1/len(collection)
        model_label = row.label_pred
        count = row.iou  # This how many of this label model annotations overlap with this label human annotation
        if model_label < 0:
            HTP[user_label] += count / len(collections[model_label])
            for conceptid in collections[model_label]:
                if conceptid == user_label:
                    continue
                HFP[conceptid] += count / len(collections[model_label])
        else:
            HTP[user_label] += count
            TP[user_label] += count
    return pd.Series(HTP), pd.Series(HFP), pd.Series(TP)


def get_count(count_values, concept):
    return count_values[concept] if concept in count_values.index else 0


def get_precision(TP, FP):
    return TP / (TP + FP) if (TP + FP) != 0 else 0


def get_recall(TP, FN):
    return TP / (TP + FN) if (TP + FN) != 0 else 0


def get_f1(recall, precision):
    return (2 * recall * precision / (precision + recall)) if (precision + recall) != 0 else 0


def count_accuracy(true_num, pred_num):
    if true_num == 0:
        return 1.0 if pred_num == 0 else 0
    else:
        return 1 - (abs(true_num - pred_num) / max(true_num, pred_num))


def get_recall_precision_f1_counts(TP, FP, FN):
    pred_num, true_num = TP+FP, TP+FN
    r, p = get_recall(TP, FN), get_precision(TP, FP)
    return p, r, get_f1(r, p), pred_num, true_num, count_accuracy(true_num, pred_num)


def generate_metrics(concepts, list_of_classifications):
    metrics = pd.DataFrame()
    for concept in concepts:
        HTP, HFP, HFN, TP, FP, FN = [
            get_count(classification, concept) for classification in list_of_classifications]

        metrics = metrics.append([
            [
                concept,
                HTP, HFP, HFN, *get_recall_precision_f1_counts(HTP, HFP, HFN),
                TP, FP, FN, *get_recall_precision_f1_counts(TP, FP, FN)
            ]
        ])
    metrics.columns = [
        "conceptid",
        "H_TP", "H_FP", "H_FN", "H_Precision", "H_Recall", "H_F1", "H_pred_num", "H_true_num", "H_count_accuracy",
        "TP", "FP", "FN", "Precision", "Recall", "F1", "pred_num", "true_num", "count_accuracy"]
    return metrics.sort_values(by='conceptid')


def get_human_annotations(validation, correctly_classified_objects):
    human_annotations = correctly_classified_objects[
        ['x1_val', 'y1_val', 'x2_val', 'y2_val', 'label_val', 'userid', 'originalid', 'frame_num']]
    human_annotations = human_annotations.rename(
        columns={'x1_val': 'x1', 'y1_val': 'y1', 'x2_val': 'x2', 'y2_val': 'y2', 'label_val': 'label'})
    human_annotations = validation[~validation.originalid.isin(
        correctly_classified_objects.originalid)].drop_duplicates(subset='originalid').append(human_annotations)
    return human_annotations

# Some videos contain no conepts, so we set their count to be default


def check_for_all_concepts(value_counts, concepts):
    if value_counts.empty:
        value_counts = pd.DataFrame({})
    for conceptid in concepts:
        if conceptid not in value_counts.index:
            value_counts[conceptid] = 0


def score_predictions(validation, predictions, iou_thresh, concepts, collections):
    cords = ['x1', 'y1', 'x2', 'y2']
    val_suffix = '_val'
    pred_suffix = '_pred'

    # Match human and model annotations by frame number
    merged_user_pred_annotations = pd.merge(
        validation,
        predictions,
        suffixes=[val_suffix, pred_suffix],
        on='frame_num')
    # Only keep rows which the predicted label matching validation (or collection)
    merged_user_pred_annotations = merged_user_pred_annotations[merged_user_pred_annotations.apply(
        lambda row: True if row.label_val == row.label_pred or (row.label_pred < 0 and row.label_val in collections[row.label_pred]) else False, axis=1)]

    # get data from validation x_val...
    merged_val_x_y = merged_user_pred_annotations[[cord + val_suffix for cord in cords]].to_numpy()
    # get data for pred data x_pred...
    merged_pred_x_y = merged_user_pred_annotations[[cord + pred_suffix for cord in cords]].to_numpy()

    # Get iou for each row
    iou = vectorized_iou(merged_val_x_y, merged_pred_x_y)
    merged_user_pred_annotations = merged_user_pred_annotations.assign(iou=iou)

    # Correctly Classified must have iou greater than or equal to threshold
    max_iou = merged_user_pred_annotations.groupby("originalid").iou.max().to_frame().reset_index()
    max_iou = max_iou[max_iou["iou"] >= iou_thresh]
    correctly_classified_objects = pd.merge(
        merged_user_pred_annotations,
        max_iou,
        how="inner",
        left_on=["originalid", "iou"],
        right_on=["originalid", "iou"]
    ).drop_duplicates(subset='objectid')

    #    Positive
    pred_objects_no_val = predictions[~predictions.objectid.isin(
        correctly_classified_objects.objectid)].drop_duplicates(subset='objectid')
    HFP = pred_objects_no_val['label'].value_counts()
    check_for_all_concepts(HFP, concepts)
    HFP, FP = convert_hierarchy_fp_counts(HFP, collections)

    # True Positive
    pred_val_label_counts = correctly_classified_objects.groupby(["label_pred", "label_val"])["iou"].count()
    HTP, HFP, TP = convert_hierarchy_tp_counts(pred_val_label_counts, HFP, collections, concepts)

    # False Negative
    HFN = validation[~validation.originalid.isin(correctly_classified_objects.originalid)].drop_duplicates(
        subset='originalid').label.value_counts()
    check_for_all_concepts(HFN, concepts)
    FN = validation[~validation.originalid.isin(correctly_classified_objects[correctly_classified_objects.label_pred > 0].originalid)].drop_duplicates(
        subset='originalid').label.value_counts()
    check_for_all_concepts(FN, concepts)

    return generate_metrics(
        concepts, [HTP, HFP, HFN, TP, FP, FN]), get_human_annotations(
        validation, correctly_classified_objects)


def update_ai_videos_database(model_username, video_id, filename, local_con=None):
    # Get the model's name
    username_split = model_username.split('-')
    version = username_split[-1]
    model_name = '-'.join(username_split[:-1])

    # add the entry to ai_videos
    cursor = local_con.cursor()
    cursor.execute('''
            INSERT INTO ai_videos (name, videoid, version, model_name)
            VALUES (%s, %s, %s, %s)''',
                   (filename, video_id, version, model_name)
                   )
    local_con.commit()


def save_metrics(metrics, video_id, model_name, start_frame=None, end_frame=None):
    file_name = f"./{model_name}_{str(video_id)}"
    if start_frame != None:
        print("testing1")
        file_name += f"_{start_frame}"
    if end_frame != None:
        print("testing2")
        file_name += f"_{end_frame}"
    metrics.to_csv(f"{file_name}.csv")


def get_video_capture(video_filename):
    vid_filepath = os.path.join(config.VIDEO_FOLDER, video_filename)

    if not os.path.exists(vid_filepath):
        print("Video filename not found")

    vid = cv2.VideoCapture(vid_filepath)
    while not vid.isOpened():
        time.sleep(1)

    print(f'total frames {vid.get(cv2.CAP_PROP_FRAME_COUNT)}')
    return vid


def save_video(filename, s3=None):
    # convert to mp4 and upload to s3 and db
    # requires temp so original not overwritten
    converted_file = str(uuid.uuid4()) + ".mp4"
    # Convert file so we can stream on s3
    ff = FFmpeg(
        inputs={filename: ['-loglevel', '0']},
        outputs={converted_file: ['-codec:v', 'libx264', '-y']}
    )
    print(ff.cmd)
    print(psutil.virtual_memory())
    ff.run()

    # temp = ['ffmpeg', '-loglevel', '0', '-i', filename,
    #         '-codec:v', 'libx264', '-y', converted_file]
    # subprocess.call(temp)
    # upload video..
    s3.upload_file(
        converted_file, config.S3_BUCKET,
        config.S3_BUCKET_AIVIDEOS_FOLDER + filename,
        ExtraArgs={'ContentType': 'video/mp4'})
    # remove files once uploaded
    os.system('rm \'' + filename + '\'')
    os.system('rm ' + converted_file)

    cv2.destroyAllWindows()

# Generates the video with the ground truth frames interlaced


def generate_video(filename, video_capture, results, concepts, video_id, annotations):
    print("Inside generating video")
    # Combine human and prediction annotations
    results = results.append(annotations, sort=True)
    # Cast frame_num to int (prevent indexing errors)
    results.frame_num = results.frame_num.astype('int')
    classmap = get_classmap(concepts)

    # make a dictionary mapping conceptid to count (init 0)
    conceptsCounts = {concept: 0 for concept in concepts}
    total_length = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
    seenObjects = {}

    print("Opening video writer")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(config.AI_VIDEO_FOLDER + "/" + filename, fourcc, video_capture.get(cv2.CAP_PROP_FPS),
                          (config.RESIZED_WIDTH, config.RESIZED_HEIGHT))
    print("Opened video writer")

    video_capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
    for frame_num in range(total_length):
        check, frame = video_capture.read()
        if not check:
            break

        for res in results[results.frame_num == frame_num].itertuples():
            x1, y1, x2, y2 = int(res.x1), int(res.y1), int(res.x2), int(res.y2)
            # boxText init to concept name
            boxText = classmap[concepts.index(res.label)]

            if pd.isna(res.confidence):  # No confidence means user annotation
                # Draws a (user) red box
                # Note: opencv uses color as BGR
                cv2.rectangle(frame, (x1, y1),
                              (x2, y2), (0, 0, 255), 2)
            else:  # if confidence exists -> AI annotation
                # Keeps count of concepts
                if (res.objectid not in seenObjects):
                    conceptsCounts[res.label] += 1
                    seenObjects[res.objectid] = conceptsCounts[res.label]
                # Draw an (AI) green box
                cv2.rectangle(frame, (x1, y1),
                              (x2, y2), (0, 255, 0), 2)
                # boxText = count concept-name (confidence) e.g. "1 Starfish (0.5)"
                boxText = str(seenObjects[res.objectid]) + " " + boxText + \
                    " (" + str(round(res.confidence, 3)) + ")"
            cv2.putText(
                frame, boxText,
                (x1 - 5, y2 + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        out.write(frame)

    out.release()


def get_validation_set(video_fps, video_id, concepts, tracking_id, start_frame=None, end_frame=None):
    good_users = config.GOOD_USERS
    good_users.append(tracking_id)
    data = pd.read_csv(config.DATABASE_FOLDER + '/annotations.csv', low_memory=False)

    # get biologist annotations from the video
    print('get biology')
    video_annotations = data[(data.videoid == video_id) & (
        data.userid.isin(good_users)) & (data.conceptid.isin(concepts))]

    # calculate annotation's frame number
    print('calc frame')
    video_annotations.loc[pd.isna(video_annotations.framenum), 'frame_num'] = video_annotations.loc[pd.isna(
        video_annotations.framenum), 'timeinvideo'] * video_fps
    video_annotations.loc[~pd.isna(video_annotations.framenum), 'frame_num'] = video_annotations.loc[~pd.isna(
        video_annotations.framenum), 'framenum']
    # reformat
    print('reformat')
    video_annotations = video_annotations.astype({'frame_num': 'int'})
    video_annotations = video_annotations[(video_annotations.frame_num <= end_frame) &
                                          ((video_annotations.frame_num >= start_frame))]
    validation = video_annotations[['x1', 'y1', 'x2', 'y2', 'conceptid',
                                    'videowidth', 'videoheight', 'userid', 'originalid', 'frame_num']]
    validation = validation.rename(columns={'conceptid': 'label'})
    print('resizing')
    validation['x1'] = validation['x1'] * config.RESIZED_WIDTH / validation['videowidth']
    validation['x2'] = validation['x2'] * config.RESIZED_WIDTH / validation['videowidth']
    validation['y1'] = validation['y1'] * config.RESIZED_HEIGHT / validation['videoheight']
    validation['y2'] = validation['y2'] * config.RESIZED_HEIGHT / validation['videoheight']
    print('Done Calculating resizing x,y')
    validation = validation.drop(['videowidth', 'videoheight'], axis=1)
    return validation


def create_annotation_collection(model_name, user_id, video_id, concept_ids, upload_annotations, local_con=None):
    if not upload_annotations:
        raise ValueError("cannot create new annotation collection if "
                         "annotations aren't uploaded")
    if user_id is None:
        raise ValueError("user_id is None, cannot create new collection")

    time_now = datetime.datetime.now().strftime(r"%y-%m-%d_%H:%M:%S")
    collection_name = '_'.join([model_name, str(video_id), time_now])
    description = f"By {model_name} on video {video_id} at {time_now}"

    concept_names = pd_query(
        """
        SELECT name
        FROM concepts
        WHERE id IN %s
        """, params=(tuple(concept_ids),), local_con=local_con
    )['name'].tolist()
    string_conceptids = str(concept_ids)
    string_conceptids = string_conceptids.replace('(', '{')
    string_conceptids = string_conceptids.replace(')', '}')
    print(string_conceptids)
    cursor = local_con.cursor()
    cursor.execute(
        """
        INSERT INTO annotation_collection
        (name, description, users, videos, concepts, tracking, conceptid)
        VALUES (%s, %s, %s, %s, %s, %s, %s::integer[])
        RETURNING id
        """,
        (collection_name, description, [user_id], [video_id], concept_names,
            False, string_conceptids)
    )
    local_con.commit()
    collection_id = int(cursor.fetchone()[0])

    return collection_id
