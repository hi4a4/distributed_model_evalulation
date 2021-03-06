import cv2
from psycopg2 import connect
import boto3
import os
from dotenv import load_dotenv
import datetime
import copy
import time
import uuid
import sys
import math
import json
import subprocess
from PIL import Image
import numpy as np
from itertools import zip_longest
from skimage.measure import compare_ssim

from config.config import RESIZED_WIDTH, RESIZED_HEIGHT, S3_BUCKET, \
    S3_ANNOTATION_FOLDER, S3_VIDEO_FOLDER, DB_NAME, DB_USER, DB_PASSWORD, \
    DB_HOST, AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, LENGTH
s3 = boto3.client(
    's3',
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY)

# initialize a dictionary that maps strings to their corresponding
# OpenCV object tracker implementations
OPENCV_OBJECT_TRACKERS = {
    "csrt": cv2.TrackerCSRT_create,
    "kcf": cv2.TrackerKCF_create,
    "boosting": cv2.TrackerBoosting_create,
    "mil": cv2.TrackerMIL_create,
    "tld": cv2.TrackerTLD_create,
    "medianflow": cv2.TrackerMedianFlow_create,
    "mosse": cv2.TrackerMOSSE_create
}


def getS3Image(image):
    try:
        obj = s3.get_object(
            Bucket=S3_BUCKET,
            Key=S3_ANNOTATION_FOLDER + image
        )
    except:
        print("Annotation missing image: " + image)
        return
    # Get image in RGB and transform to BGR
    img = Image.open(obj['Body'])
    img = np.asarray(img)
    img = img[:, :, :3]
    img = img[:, :, ::-1]
    # Resize to video width/height
    img = cv2.resize(img, (RESIZED_WIDTH, RESIZED_HEIGHT))
    return img


def getTrackingUserid(cursor):
    cursor.execute("SELECT id FROM users WHERE username=%s", ("tracking",))
    return cursor.fetchone()[0]


def getVideoURL(cursor, videoid):
    """
    Returns
        url - video's secure streaming url
    """
    cursor.execute("SELECT filename FROM videos WHERE id=%s",
                   (str(videoid),))

    # grab video stream
    url = s3.generate_presigned_url('get_object',
                                    Params={'Bucket': S3_BUCKET,
                                            'Key': S3_VIDEO_FOLDER + cursor.fetchone()[0]},
                                    ExpiresIn=100)
    return url


def getVideoFrames(url, start, end):
    """
    Returns
        fps - frames per second of video
        frame_list - frames before timeinvideo
        frame_num - start time's frame number
    """
    cap = cv2.VideoCapture(url)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.set(0, start)  # tell video to start at 'start' time
    frame_num = (int(cap.get(1)))  # get frame number
    check = True
    frame_list = []
    curr = start

    while (check and curr <= end):
        check, frame = cap.read()
        if check:
            frame_list.append(
                cv2.resize(frame, (RESIZED_WIDTH, RESIZED_HEIGHT)))
        curr = cap.get(0)  # get time in milliseconds
    cap.release()
    return frame_list, fps, frame_num


def upload_image(frame_num, frame,
                 id, videoid, conceptid, comment, unsure,
                 x1, y1, x2, y2, cursor, con, TRACKING_ID, timeinvideo):
    # Uploads images and puts annotation in database
    image_name = str(videoid) + "_" + str(timeinvideo) + "_tracking.png"
    temp_file = str(uuid.uuid4()) + ".png"
    cv2.imwrite(temp_file, frame)
    s3.upload_file(temp_file, S3_BUCKET, S3_ANNOTATION_FOLDER +
                   image_name, ExtraArgs={'ContentType': 'image/png'})
    os.system('rm ' + temp_file)
    cursor.execute(
        """
     INSERT INTO annotations (
     framenum, videoid, userid, conceptid, timeinvideo, x1, y1, x2, y2,
     videowidth, videoheight, dateannotated, image, comment, unsure, originalid)
     VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
      """,
        (
            frame_num, videoid, TRACKING_ID, conceptid, timeinvideo, x1, y1,
            x2, y2, RESIZED_WIDTH, RESIZED_HEIGHT, datetime.datetime.now().date(), image_name,
            comment, unsure, id
        )
    )
    con.commit()
    return


def upload_video(priorFrames, postFrames, id):
    completed = False
    # Order priorFrames by time
    priorFrames.reverse()
    # Combine all frames
    priorFrames.extend(postFrames)

    output_file = str(uuid.uuid4()) + ".mp4"
    converted_file = str(uuid.uuid4()) + ".mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_file, fourcc, 20,
                          (RESIZED_WIDTH, RESIZED_HEIGHT))
    for frame in priorFrames:
        out.write(frame)
    out.release()
    # Convert file so we can stream on s3
    temp = ['ffmpeg', '-loglevel', '0', '-i', output_file,
            '-codec:v', 'libx264', '-y', converted_file]
    subprocess.call(temp)

    if os.path.isfile(converted_file):
        # upload video..
        s3.upload_file(
            converted_file,
            S3_BUCKET,
            S3_VIDEO_FOLDER + str(id) + "_tracking.mp4",
            ExtraArgs={'ContentType': 'video/mp4'}
        )
        os.system('rm ' + converted_file)
        completed = True
    else:
        print("Failed to make video for annotations: " + str(id))
    os.system('rm ' + output_file)
    return completed


def matchS3Frame(priorFrames, postFrames, s3Image):
    best_score = 0
    best_index = None
    for index, (prior, post) in enumerate(
            zip_longest(reversed(priorFrames), postFrames)):
        if prior is not None:
            (prior_score, _) = compare_ssim(
                s3Image, prior, full=True, multichannel=True)
            if prior_score > best_score:
                best_score = prior_score
                best_index = -index
        if post is not None:
            (post_score, _) = compare_ssim(
                s3Image, post, full=True, multichannel=True)
            if post_score > best_score:
                best_score = post_score
                best_index = index
    return best_score, best_index


def fix_offset(priorFrames, postFrames, s3Image, fps, timeinvideo,
               frame_num, id, cursor, con):
    best_score, best_index = matchS3Frame(priorFrames, postFrames, s3Image)
    if best_index == 0:
        # No change necessary
        return priorFrames, postFrames, timeinvideo, frame_num
    elif best_score > .9:
        timeinvideo = round(timeinvideo + (best_index / fps), 2)
        frame_num = frame_num + best_index
        cursor.execute(
            '''
                UPDATE annotations
                SET framenum=%s, timeinvideo=%s, originalid=NULL
                WHERE id= %s;
            ''',
            (frame_num, timeinvideo, id))
        con.commit()
    else:
        print(
            f'Failed on annnotation {id} with best score {best_score}')
        cursor.execute(
            "UPDATE annotations SET unsure=TRUE WHERE id=%s;", (id,))
        con.commit()
    if best_index > 0:
        tempFrames = postFrames[:best_index + 1]
        priorFrames = priorFrames + tempFrames
        del postFrames[:best_index]
    else:
        tempFrames = priorFrames[best_index - 1:]
        postFrames = tempFrames + postFrames
        del priorFrames[best_index:]
    return priorFrames, postFrames, timeinvideo, frame_num


def track_object(frame_num, frames, box, track_forward, end,
                 id, videoid, conceptid, comment, unsure,
                 cursor, con, TRACKING_ID, fps, timeinvideo):
    # Tracks the object forwards and backwards in a video
    frame_list = []
    time_elapsed = 0
    trackers = cv2.MultiTracker_create()
    # initialize tracking, add first frame (original annotation)
    tracker = OPENCV_OBJECT_TRACKERS["kcf"]()

    # keep tracking object until its out of frame or time is up
    for index, frame in enumerate(frames):
        (x1, y1, w, h) = [int(v) for v in box]
        x2 = x1 + w
        y2 = y1 + h
        # Remove invalid bounding boxes
        if (
                x1 > RESIZED_WIDTH or
                y1 > RESIZED_HEIGHT or
                x2 < 0 or
                y2 < 0 or
                x1 == x2 or
                y1 == y2):
            continue
        time_elapsed += (1 / fps) if track_forward else - (1 / fps)
        frame_num += 1 if track_forward else -1
        if index == 0:  # initialize bounding box in first frame
            trackers.add(tracker, frame, box)
        (success, boxes) = trackers.update(frame)
        if success:
            box = boxes[0]
            (x1, y1, w, h) = [int(v) for v in box]
            x2 = x1 + w
            y2 = y1 + h
            # Fix box if outside video frame
            x1 = x1 if x1 > 0 else 0
            y1 = y1 if y1 > 0 else 0
            x2 = x2 if x2 < RESIZED_WIDTH else RESIZED_WIDTH
            y2 = y2 if y2 < RESIZED_HEIGHT else RESIZED_HEIGHT
            # Remove invalid bounding boxes
            if (
                    x1 > RESIZED_WIDTH or
                    y1 > RESIZED_HEIGHT or
                    x2 < 0 or
                    y2 < 0 or
                    x1 == x2 or
                    y1 == y2 or
                    (y2-y1) * (x2-x1) < 1):
                continue
            
            if index != 0:
                upload_image(frame_num, frame, id, videoid,
                             conceptid, comment, unsure,
                             x1, y1, x2, y2,
                             cursor, con, TRACKING_ID,
                             round(timeinvideo + time_elapsed, 2))
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        frame_list.append(frame)
    cv2.destroyAllWindows()
    return frame_list


def track_annotation(id, conceptid, timeinvideo, videoid, image,
                     videowidth, videoheight, x1, y1, x2, y2, comment, unsure):
    print("Start tracking annotation: " + str(id))
    con = connect(database=DB_NAME, host=DB_HOST,
                  user=DB_USER, password=DB_PASSWORD)
    cursor = con.cursor()

    # Make bounding box adjusted to video width and height
    x_ratio = (videowidth / RESIZED_WIDTH)
    y_ratio = (videoheight / RESIZED_HEIGHT)
    x1 = x1 / x_ratio
    y1 = y1 / y_ratio
    width = (x2 / x_ratio) - x1
    height = (y2 / y_ratio) - y1
    box = (x1, y1, width, height)

    TRACKING_ID = getTrackingUserid(cursor)
    url = getVideoURL(cursor, videoid)
    s3Image = getS3Image(image)
    if s3Image is None:
        return False

    # initialize video for grabbing frames before annotation
    # start vidlen/2 secs before obj appears
    start = ((timeinvideo * 1000) - (LENGTH / 2))
    end = start + (LENGTH / 2)  # end when annotation occurs
    # Get frames tracking_vid_length/2 before timeinvideo
    # Note: if annotation timeinvideo=0 -> priorFrames = []
    priorFrames, _, _ = getVideoFrames(url, start, end)

    # initialize vars for getting frames post annotation
    start = timeinvideo * 1000
    end = start + (LENGTH / 2)
    postFrames, fps, frame_num = getVideoFrames(url, start, end)

    # Fix weird javascript video currentTime randomization
    priorFrames, postFrames, timeinvideo, frame_num = fix_offset(
        priorFrames, postFrames, s3Image, fps,
        timeinvideo, frame_num, id, cursor, con)

    # tracking forwards..
    postFrames = track_object(
        frame_num, postFrames, box, True, end,
        id, videoid, conceptid, comment, unsure,
        cursor, con, TRACKING_ID, fps, timeinvideo)

    # tracking backwards
    priorFrames = track_object(
        frame_num, reversed(priorFrames), box, False, 0,
        id, videoid, conceptid, comment, unsure,
        cursor, con, TRACKING_ID, fps, timeinvideo)

    upload_video(priorFrames, postFrames, id)

    cv2.destroyAllWindows()
    con.close()
    print("Done tracking annotation: " + str(id))
    return True
