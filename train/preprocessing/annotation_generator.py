import os
import sys
import time
import random
from collections import OrderedDict

import boto3
import pandas as pd
import numpy as np
from six import raise_from
from PIL import Image, ImageFile, ImageDraw
from botocore.exceptions import ClientError
from keras_retinanet.preprocessing.csv_generator import Generator
from keras_retinanet.utils.image import read_image_bgr, resize_image

from utils.query import pd_query, cursor
from config import config


# Without this the program will crash
ImageFile.LOAD_TRUNCATED_IMAGES = True


def error_print(message):
    print(f'[Warning] {message}', file=sys.stderr)


def _parse(value, function, fmt):
    """
    Parse a string into a value, and format a nice ValueError if it fails.
    Returns `function(value)`.
    Any `ValueError` raised is catched and a new `ValueError` is raised
    with message `fmt.format(e)`, where `e` is the caught `ValueError`.
    """
    try:
        return function(value)
    except ValueError as e:
        raise_from(ValueError(fmt.format(e)), None)


def _atomic_file_exists(file_path):
    """
    Atomically check if a file exists
    Returns a boolean representing if the file exists
    """
    try:
        # This file open is atomic. This avoids race conditions when multiple processes are running.
        # This race condition only happens when workers > 1 and multiprocessing = True in fit_generator
        fd = os.open(file_path, os.O_CREAT | os.O_EXCL)
        os.close(fd)
        return False
    except FileExistsError:
        return True


def _get_labelmap(classes):
    """
    Initializes the classmap of each class's database IDs to training IDs
    """

    # Keras requires that the mapping IDs correspond to the index number of the class.
    # So we create that mapping (dictionary)
    class_id_name = pd_query(
        f"select id, name from concepts where id = ANY(ARRAY{classes})")
    labelmap = pd.Series(class_id_name.name.values,
                         index=class_id_name.id).to_dict()

    return labelmap


def get_classmap(classes):
    """
    Initializes the classmap of each class's database IDs to training IDs
    """

    # Keras requires that the mapping IDs correspond to the index number of the class.
    # So we create that mapping (dictionary)
    classmap = {class_: index for index, class_ in enumerate(classes)}

    return classmap


def _bound_coordinates(curr):
    x_ratio = (curr['videowidth'] / config.RESIZED_WIDTH)
    y_ratio = (curr['videoheight'] / config.RESIZED_HEIGHT)

    x1 = min(max(int(curr['x1'] / x_ratio), 0), config.RESIZED_WIDTH)
    x2 = min(max(int(curr['x2'] / x_ratio), 0), config.RESIZED_WIDTH)

    y1 = min(max(int(curr['y1'] / y_ratio), 0), config.RESIZED_HEIGHT)
    y2 = min(max(int(curr['y2'] / y_ratio), 0), config.RESIZED_HEIGHT)

    return x1, x2, y1, y2


class AnnotationGenerator(object):

    def __init__(self,
                 collection_ids,
                 verified_only,
                 include_tracking,
                 verify_videos,
                 classes,
                 min_examples,
                 validation_split=0.8):

        print("Grabbing annotations....")
        # Start with a list of all possible annotations, grouped by frame in video
        selected_frames, concept_counts = self._select_annotations(collection_ids,
                                                                   verified_only,
                                                                   include_tracking,
                                                                   verify_videos,
                                                                   min_examples,
                                                                   classes)
        print(f"Found {sum(concept_counts.values())} annotations.")
        # Creating a counter for each user: the concepts & whether they were verified.
        userDict = {}
        for user_df in selected_frames:
            userId = str(user_df['userid'].iloc[0])
            if userId not in userDict:
                userDict[userId] = {}
            for index,row in user_df.iterrows(): # each unique concept
                conceptId = str(row['conceptid'])
                if conceptId not in userDict[userId]:
                    userDict[userId][conceptId] = {'0': 0, '1':0} # 0 is un-verified, 1 is verified
                if pd.notnull(row['verifiedby']): # it has been verified, increment verified count
                    userDict[userId][conceptId]['1'] +=1
                else:
                    userDict[userId][conceptId]['0'] +=1
        self.userDict = userDict
        self.selected_frames = selected_frames
        self.classmap = get_classmap(classes)

        # Shuffle selected frames so that training/testing set are different each run
        random.shuffle(self.selected_frames)

        num_frames = len(self.selected_frames)
        split_index = int(num_frames * validation_split)

        # Split our data into training and testing sets, based on validation_splitppp
        self.training_set = self.selected_frames[0:split_index]
        self.testing_set = self.selected_frames[split_index:]

    def flow_from_s3(self,
                     image_folder='',
                     image_extension='.png',
                     subset='training',
                     **kwargs):

        # Depending on subset, return either a training or testing generator
        if subset == 'training':
            return S3Generator(
                selected_frames=self.training_set,
                image_folder=image_folder,
                image_extension=image_extension,
                classes=self.classmap,
                **kwargs
            )
        elif subset in ['validation', 'testing']:
            return S3Generator(
                selected_frames=self.testing_set,
                image_folder=image_folder,
                image_extension=image_extension,
                classes=self.classmap,
                **kwargs
            )
        else:
            raise ValueError(
                'subset parameter must be either "training" or "validation"/"testing"')

    @staticmethod
    def _select_annotations(collection_ids, verified_only, include_tracking, verify_videos, min_examples, concepts):
        selected = []
        concept_count = {}

        annotations = AnnotationGenerator._get_annotations(
            collection_ids, verified_only, include_tracking,
            verify_videos, concepts, min_examples)

        for concept in concepts:
            concept_count[concept] = 0

        # This grouping ensure that we can view all annotations for a single image
        frames_group = annotations.groupby(
            ['videoid', 'frame_num'], sort=False)
        frames_group = [df for _, df in frames_group]

        ai_id = pd_query(
            "SELECT id FROM users WHERE username='tracking'").id[0]

        # Give priority to frames with highest verification priority
        # And with least amount of tracking annotations
        # And lower speed

        frames_group.sort(key=lambda df: (
            -df.priority.max(), list(df['userid']).count(ai_id), df.speed.mean()))

        # Selects images that we'll use (each group has annotations for an image)
        for annotation_group in frames_group:
            # Check if we have min number of images already
            if not any(v < min_examples for v in concept_count.values()):
                break

            in_annot = []
            for i, row in annotation_group.iterrows():
                if row['conceptid'] not in concept_count:
                    continue
                concept_count[row['conceptid']] += 1
                in_annot.append(row['conceptid'])

                x1, x2, y1, y2 = _bound_coordinates(row)

                annotation_group.at[i, 'x1'] = x1
                annotation_group.at[i, 'x2'] = x2
                annotation_group.at[i, 'y1'] = y1
                annotation_group.at[i, 'y2'] = y2
                annotation_group.at[i, 'videowidth'] = config.RESIZED_WIDTH
                annotation_group.at[i, 'videoheight'] = config.RESIZED_HEIGHT

            # Checks if frame has only concept we have too many of
            if any(v > min_examples for v in concept_count.values()):
                # Gets all concepts that we have too many of
                excess = list(
                    {key: value for (key, value) in concept_count.items() if value > min_examples})

                # if it doens't include concept that we need more of
                # Don't save the annotation
                if set(excess) >= set(in_annot):
                    for a in in_annot:
                        concept_count[a] -= 1
                    continue
            selected.append(annotation_group)

        return selected, concept_count

    @staticmethod
    def _get_annotations(collection_ids, verified_only, include_tracking,
                         verify_videos, concepts, min_examples):
        # Query that gets all annotations for given concepts
        # making sure that any tracking annotations originated from good users
        cursor.execute(
            """SELECT id FROM users WHERE username = 'tracking'""")
        tracking_uid = cursor.fetchone()[0]

        cursor.execute(
                    """SELECT fps FROM videos LIMIT 1""")
        fps = cursor.fetchone()[0]

        annotations_query = f'''
            WITH collection AS (SELECT
                A.id,
                image,
                userid,
                videoid,
                videowidth,
                videoheight,
                conceptid,
                x1, x2, y1, y2,
                speed,
                CASE WHEN
                    framenum is not null
                THEN
                    framenum
                ELSE
                    FLOOR(timeinvideo*{fps})
                END AS frame_num,
                verifiedby
            FROM
                annotation_intermediate inter
            LEFT JOIN
                annotations a ON a.id=inter.annotationid
            LEFT JOIN
                videos ON videos.id=videoid
            WHERE inter.id = ANY(%s) AND a.videoid <> ALL(%s)
        '''

        if verified_only:
            annotations_query += """ AND a.verifiedby IS NOT NULL"""

        # Filter collection so each concept has min_example annotations
        annotations_query += f'''
            ),
            filteredCollection AS (
                SELECT
                    *
                FROM (
                    SELECT
                        ROW_NUMBER() OVER (
                            PARTITION BY
                                conceptid
                            ORDER BY userid={tracking_uid}, verifiedby IS NULL) AS r,
                        c.*
                    FROM
                        collection c) t
                WHERE
                    t.r <= (%s)
            )
        '''
        # Add annotations that exist in the same frame
        annotations_query += f'''
            SELECT
                A.id,
                image,
                userid,
                videoid,
                videowidth,
                videoheight,
                conceptid,
                x1, x2, y1, y2,
                speed,
                priority,
                CASE WHEN
                    framenum is not null
                THEN
                    framenum
                ELSE
                    FLOOR(timeinvideo*{fps})
                END AS frame_num,
                verifiedby
            FROM
                annotations a
            LEFT JOIN
                videos ON videos.id=videoid
            WHERE
                ABS(x2-x1)>25 AND ABS(y2-y1)>25 AND
                x1>=0 AND x1<videowidth AND
                x2>0 AND x2<=videowidth AND
                y1>=0 AND y1<videowidth AND
                y2>0 AND y2<=videowidth AND
                (a.userid = {tracking_uid} OR 
                a.userid in {str(tuple(config.GOOD_USERS))} OR
                a.verifiedby is not null) AND
                EXISTS (
                    SELECT
                        1
                    FROM
                        filteredCollection c
                    WHERE
                        c.videoid=a.videoid
                        AND c.frame_num=ROUND(fps * timeinvideo))
        '''
        if not include_tracking:
            annotations_query += f''' AND a.userid <> {tracking_uid}'''

        return pd_query(
            annotations_query,
            (collection_ids, verify_videos, min_examples))


class S3Generator(Generator):

    def __init__(self, classes, selected_frames, image_folder, image_extension='.png', **kwargs):

        self.image_folder = image_folder

        # We initalize selected_annotations to hold all possible annotation iamges.
        # Then, downloaded_images will hold those that have already been downloaded
        self.selected_annotations = pd.concat(selected_frames).reset_index()

        # Go ahead and add a column with the file name we'll save the images as
        # We use videoid + frame_num as this ensures that we never download
        # the same frame in a video twice, even if it has multiple annotations
        self.selected_annotations['save_name'] = self.selected_annotations.apply(
            lambda row: f'{row["videoid"]}_{int(row["frame_num"])}',
            axis=1
        )

        # Make a set of all images that've already been downloaded
        self.downloaded_images = set(os.listdir(image_folder))

        self.image_extension = image_extension
        self.classes = classes

        self.labelmap = _get_labelmap(list(classes))
        self.failed_downloads = set()

        # Make a reverse dictionary so that we can lookup the other way
        self.labels = {}
        for key, value in self.classes.items():
            self.labels[value] = key

        self._connect_s3()

        self.image_data = self._read_annotations()
        super(S3Generator, self).__init__(**kwargs)

    def size(self):
        """ Size of the dataset.
        """
        return len(self.selected_annotations.index)

    def num_classes(self):
        """ Number of classes in the dataset.
        """
        return len(self.classes)

    def has_label(self, label):
        """ Return True if label is a known label.
        """
        return label in self.labels

    def has_name(self, name):
        """ Returns True if name is a known class.
        """
        return name in self.classes

    def name_to_label(self, name):
        """ Map name to label.
        """
        return self.classes[name]

    def label_to_name(self, label):
        """ Map label to name.
        """
        return self.labelmap[self.labels[label]]

    def image_aspect_ratio(self, image_index):
        """ Compute the aspect ratio for an image with image_index.
        """
        image = self.selected_annotations.iloc[image_index]
        image_width = image['videowidth']
        image_height = image['videoheight']

        return float(image_width) / float(image_height)

    def load_image(self, image_index):
        """ Load an image at the image_index.
        """
        if self._download_image(image_index):
            return read_image_bgr(self.image_path(image_index))
        return self.load_image((image_index + 1) % self.size())

    def image_path(self, image_index):
        """ Returns the image path for image_index.
        """
        image = self.selected_annotations.iloc[image_index]
        image_name = f"{image['save_name']}_{config.RESIZED_WIDTH}_{config.RESIZED_HEIGHT}"
        return os.path.join(self.image_folder, image_name + self.image_extension)

    def load_annotations(self, image_index):
        """ Load annotations for an image_index.
        """

        # If we couldn't download this image, go onto the next one
        if image_index in self.failed_downloads:
            return self.load_annotations((image_index + 1) % self.size())

        image = self.selected_annotations.iloc[image_index]
        annotations = {'labels': np.empty((0,)), 'bboxes': np.empty((0, 4))}
        image_name = image['save_name']

        # Add all bounding boxes and annotations to the annotations dict for a particular image
        for idx, annot in enumerate(self.image_data[image_name]):
            annotations['labels'] = np.concatenate(
                (annotations['labels'], [self.name_to_label(annot['class'])]))

            annotations['bboxes'] = np.concatenate((annotations['bboxes'], [[
                float(annot['x1']),
                float(annot['y1']),
                float(annot['x2']),
                float(annot['y2']),
            ]]))

        return annotations

    def _download_image(self, image_index):
        image = self.selected_annotations.iloc[image_index]
        saved_image_name = image['save_name'] + self.image_extension

        if saved_image_name in self.downloaded_images:
            return True

        image_name = str(image['image'])
        # Some files have a file extension, some don't. Let's fix that.
        if self.image_extension not in image_name:
            image_name += self.image_extension
            
        try:
            obj = self.client.get_object(
                Bucket=config.S3_BUCKET, Key=config.S3_ANNOTATION_FOLDER + image_name)
            if (obj['ContentLength'] == 0):
                print(f'{image_name} Content Length is 0')
                # Image is empty, use the next index
                error_print(
                    f'file {config.S3_ANNOTATION_FOLDER}{image_name} has size 0, using next image instead')
                self.failed_downloads.add(image_index)
                return False
            obj_image = Image.open(obj['Body'])
            resized_image = obj_image.resize(
                (config.RESIZED_WIDTH, config.RESIZED_HEIGHT))
        # ClientError is the exception class for a KeyNotFound error
        except ClientError:
            # Image doesnt exist, use the next index
            print(f'{image_name} not found')
            error_print(
                f'file {config.S3_ANNOTATION_FOLDER}{image_name} not found in S3 bucket, using next image instead')
            self.failed_downloads.add(image_index)
            return False

        

        image_path = self.image_path(image_index)

        # Atomically check if the file has already been opened.
        # If we're good, save it
        if not _atomic_file_exists(image_path):
            resized_image.save(image_path)
        else:
            # The file exists, lets make sure it's done downloading.
            # We spin while the file exists, but has no content
            while os.path.getsize(image_path) == 0:
                time.sleep(10)

        return True

    def _read_annotations(self):
        """ Read annotations from our selected annotations.
            Returns a dictionary mapping video frames (unique frame images)
            to bounding boxes and coordinates.
        """
        result = OrderedDict()

        for _, image in self.selected_annotations.iterrows():
            image_file = image['save_name']
            # We treat the database ID as the class name.
            class_name = image['conceptid']

            # We already augmented these when creating our annotations df
            # So, we can just directly assign the coordinates.
            x1, x2, y1, y2 = image[['x1', 'x2', 'y1', 'y2']].values

            if image_file not in result:
                result[image_file] = []

            # Safely parse all string coordinates into integers
            x1 = _parse(x1, int, 'malformed x1: {{}}')
            y1 = _parse(y1, int, 'malformed y1: {{}}')
            x2 = _parse(x2, int, 'malformed x2: {{}}')
            y2 = _parse(y2, int, 'malformed y2: {{}}')

            user_id = image['userid']
            user_id = _parse(user_id, float, 'bad user id')

            tracking = user_id == 32

            # Check if the current class name is a class we are predicting on
            # If not, use the image but not the annotation
            if class_name not in self.classes:
                continue

            # If a row contains only an image path, it's an image without annotations.
            if (x1, y1, x2, y2, class_name) == ('', '', '', '', ''):
                continue

            # Check that the bounding box is valid.
            if x2 <= x1:
                raise ValueError(
                    'x2 ({}) must be higher than x1 ({})'.format(x2, x1))
            if y2 <= y1:
                raise ValueError(
                    'y2 ({}) must be higher than y1 ({})'.format(y2, y1))

            result[image_file].append(
                {'x1': x1, 'x2': x2, 'y1': y1, 'y2': y2, 'class': class_name})
        return result

    def _connect_s3(self):
        aws_key = os.getenv('AWS_ACCESS_KEY_ID')
        self.client = boto3.client(
            's3',
            aws_access_key_id=aws_key,
            aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY')
        )
