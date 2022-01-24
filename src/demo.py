from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import cv2
import os
import os.path as osp
from opts import opts
from tracking_utils.utils import mkdir_if_missing
from tracking_utils.log import logger
import datasets.dataset.jde as datasets
from track import d_eval_seq

from vtrak.match_config import get_play_seconds, get_vid_name
from vtrak.config import cfg


logger.setLevel(logging.INFO)


def demo(opt):
    result_root = osp.join(cfg.output_root, 'fairmot', opt.match_name)
    mkdir_if_missing(result_root)

    opt.load_model = osp.join(cfg.checkpoints, 'FairMOT', opt.load_model)

    logger.info('Starting tracking...')
    input_video = get_vid_name(opt.match_name, opt.view)
    play_seconds = get_play_seconds(opt.match_name, opt.view,
                                    max_plays=opt.max_plays)
    dataloader = datasets.DLoadVideo(input_video, opt.img_size,
                                     play_seconds=play_seconds)
    print(f'img_size: {opt.img_size}')
    result_filename = os.path.join(result_root, f'{opt.view}.csv')
    frame_rate = dataloader.frame_rate

    if opt.output_format == 'video':
        output_video_path = osp.join(result_root, f'{opt.view}.mp4')

        vid_writer = cv2.VideoWriter(
            output_video_path, cv2.VideoWriter_fourcc(*"mp4v"),
            dataloader.frame_rate, (dataloader.w, dataloader.h)
        )
    else:
        vid_writer = None

    d_eval_seq(opt, dataloader, 'mot', result_root, opt.view, result_filename,
               vid_writer=vid_writer,
               frame_rate=frame_rate, use_cuda=opt.gpus != [-1])


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    opt = opts().init()
    demo(opt)
