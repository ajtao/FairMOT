from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths

import logging
import cv2
import os
import os.path as osp
from opts import opts
from tracking_utils.utils import mkdir_if_missing
from tracking_utils.log import logger
import datasets.dataset.jde as datasets
from track import eval_seq
import csv


logger.setLevel(logging.INFO)


def get_onlyseconds(match_csv):
    fp = open(match_csv, newline='')
    reader = csv.reader(fp, delimiter=',')

    first_line = reader.__next__()
    field2idx = {}
    for idx, field in enumerate(first_line):
        field2idx[field] = idx

    onlyseconds = {}
    in_play = False
    start_time = 0
    last_time = 0

    for idx, fields in enumerate(reader):
        # import pdb; pdb.set_trace()
        video_time = int(fields[field2idx['video_time']])
        skill = fields[field2idx['skill']]

        if skill == 'Serve':
            if in_play:
                for i in range(start_time, last_time+2):
                    onlyseconds[i] = 1
            start_time = video_time
            in_play = True

        last_time = video_time

    return onlyseconds


def demo(opt):
    if opt.match_csv is not None:
        onlyseconds = get_onlyseconds(opt.match_csv)
    else:
        onlyseconds = None

    result_root = opt.output_root if opt.output_root != '' else '.'
    mkdir_if_missing(result_root)

    logger.info('Starting tracking...')
    assert os.path.isfile(opt.input_video)
    dataloader = datasets.LoadVideo(opt.input_video, opt.img_size)
    print(f'img_size {dataloader.w}, {dataloader.h}')
    result_filename = os.path.join(result_root, 'results.txt')
    frame_rate = dataloader.frame_rate

    if opt.output_format == 'video':
        basename = osp.basename(opt.input_video)
        output_video_path = osp.join(result_root, basename)

        vid_writer = cv2.VideoWriter(
            output_video_path, cv2.VideoWriter_fourcc(*"mp4v"),
            dataloader.frame_rate, (dataloader.w, dataloader.h)
        )
    else:
        vid_writer = None

    eval_seq(opt, dataloader, 'mot', result_filename, vid_writer=vid_writer,
             frame_rate=frame_rate, use_cuda=opt.gpus!=[-1], onlyseconds=onlyseconds)


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    opt = opts().init()
    demo(opt)
