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
from track import eval_seq, d_eval_seq
import csv


logger.setLevel(logging.INFO)
                                               # vid first serve    csv
bias_table = {'Stanford_UCLA': int(1760-610),       # 29:20 = 1760     - 610
              'Stanford_Minnesota': int(2529-2287), # 42:09 = 2529     - 2287
              'Stanford_PennState': int(938-1764)}  # 15:38 = 938      - 1764

def get_onlyseconds(match_csv):
    '''
    bias to add to video_time
    '''
    bias = 0
    match_name = osp.basename(osp.dirname(match_csv))
    if match_name in bias_table:
        bias = bias_table[match_name]
        print(f'Using time bias {bias}')
    
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
        if not fields[field2idx['video_time']].isnumeric():
            continue
        video_time = int(fields[field2idx['video_time']]) + bias
        skill = fields[field2idx['skill']]
        evaluation = fields[field2idx['evaluation']]

        if skill == 'Serve':
            # Start new play
            start_time = video_time
            in_play = True

        winning = 'Winning' in evaluation
        any_error = 'Error' in evaluation
        terminal_events = [winning, any_error]

        if in_play and sum(terminal_events):
            in_play = False
            for i in range(start_time, video_time+1):
                onlyseconds[i] = 1
            
        last_time = video_time
        
    return onlyseconds


def get_play_seconds(match_csv):
    '''
    bias to add to video_time
    '''
    bias = 0
    match_name = osp.basename(osp.dirname(match_csv))
    if match_name in bias_table:
        bias = bias_table[match_name]
        print(f'Using time bias {bias}')
    
    fp = open(match_csv, newline='')
    reader = csv.reader(fp, delimiter=',')

    first_line = reader.__next__()
    field2idx = {}
    for idx, field in enumerate(first_line):
        field2idx[field] = idx

    play_seconds = []
    in_play = False
    start_time = 0
    last_time = 0

    for idx, fields in enumerate(reader):
        if not fields[field2idx['video_time']].isnumeric():
            continue
        video_time = int(fields[field2idx['video_time']]) + bias
        skill = fields[field2idx['skill']]
        evaluation = fields[field2idx['evaluation']]

        if skill == 'Serve':
            # Start new play
            start_time = video_time
            in_play = True

        winning = 'Winning' in evaluation
        any_error = 'Error' in evaluation
        terminal_events = [winning, any_error]

        if in_play and sum(terminal_events):
            in_play = False
            for i in range(start_time, video_time+1):
                play_seconds.append(i)
            
        last_time = video_time
        
    return play_seconds


def demo(opt):
    result_root = opt.output_root if opt.output_root != '' else '.'
    mkdir_if_missing(result_root)

    logger.info('Starting tracking...')
    assert os.path.isfile(opt.input_video)
    onlyseconds = None
    if opt.decord:
        play_seconds = get_play_seconds(opt.match_csv)
        dataloader = datasets.DLoadVideo(opt.input_video, opt.img_size,
                                         play_seconds=play_seconds)
    else:
        if opt.match_csv is not None:
            onlyseconds = get_onlyseconds(opt.match_csv)
        dataloader = datasets.LoadVideo(opt.input_video, opt.img_size)
    print(f'img_size {dataloader.w}, {dataloader.h}')
    result_filename = os.path.join(result_root, 'tracks.csv')
    frame_rate = dataloader.frame_rate

    if opt.output_format == 'video':
        basename = osp.splitext(osp.basename(opt.input_video))[0]
        output_video_path = osp.join(result_root, f'{basename}.mp4')

        vid_writer = cv2.VideoWriter(
            output_video_path, cv2.VideoWriter_fourcc(*"mp4v"),
            dataloader.frame_rate, (dataloader.w, dataloader.h)
        )
    else:
        vid_writer = None

    if opt.decord:
        d_eval_seq(opt, dataloader, 'mot', result_filename, vid_writer=vid_writer,
                 frame_rate=frame_rate, use_cuda=opt.gpus!=[-1])
    else:
        eval_seq(opt, dataloader, 'mot', result_filename, vid_writer=vid_writer,
                 frame_rate=frame_rate, use_cuda=opt.gpus!=[-1], onlyseconds=onlyseconds)


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    opt = opts().init()
    demo(opt)
