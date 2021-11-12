"""
This script reads a results.txt file and assembles tracklets.

We dump out cropped images that are numbered in-order for each track.
"""

import os
import csv
import cv2
import argparse
import os.path as osp

from collections import namedtuple, defaultdict
from PIL import Image


parser = argparse.ArgumentParser("ByteTrack Demo!")
parser.add_argument(
    "--tracking_csv",
    default='/home/atao/devel/FairMOT/demos/womens_clip_2plays_fairmot_dla34_0.3/results.txt', help="Tracking output file"
)
parser.add_argument(
    "--vid_fn",
    default='/home/atao/data/vball/womens_clip_2plays.mp4',
    help="Where to find image frames"
)
parser.add_argument(
    "--output_root", default='/home/atao/output/vball',
    help="output root for image crops"
)
parser.add_argument(
    "--crop_dims", default=(128,256),
    help="target resolution for player crops"
)
parser.add_argument(
    "--tracker_scale", default=1.5, type=float,
    help="how to scale tracker values down to original video resolution"
)
parser.add_argument(
    "--sampling_rate", default=15, type=int,
    help="Only sample every N frames"
)
args = parser.parse_args()

Court = namedtuple('court', ['top', 'bot', 'left', 'right'])
# half court:
court_bounds = Court(320, 680, 32, 1214)
# full court:
court_bounds = Court(193, 680, 32, 1214)
entry_fields = ['fnum', 'tid', 'x', 'y', 'w', 'h']
num_entries = len(entry_fields)
TrackEntry = namedtuple('TrackEntry', entry_fields)


def read_csv(csv_fn):
    lines = []
    fp = open(csv_fn, newline='')
    reader = csv.reader(fp, delimiter=',')
    for idx, line in enumerate(reader):
        for i in range(num_entries):
            if i<2:
                line[i] = int(line[i])
            else:
                line[i] = float(line[i]) / args.tracker_scale
        lines.append(TrackEntry._make(line[:num_entries]))
    return lines


def to_frames(lines):
    """
    Group lines into frames
    """
    frames = []
    frame = []
    fnum = lines[0].fnum
    for idx in range(1, len(lines)):
        line = lines[idx]
        if line.fnum == fnum:
            frame.append(line)
        else:
            frames.append(frame)
            frame = []
            fnum = line.fnum

    # final frame:
    frames.append(frame)
    return frames
    
        
def build_tracklets():
    """
    1. read in lines from csv
    2. build tracklets
    """
    lines = read_csv(args.tracking_csv)
    frames = to_frames(lines)
    
    tracks = {}
    tracklets = []

    for idx, frame in enumerate(frames):
        # 
        # Start new track if it doesn't exist
        # If a track existed in past but not in this frame, wrap it up
        # At EOF, wrap all tracks
        #
        existing_tids = list(tracks.keys())
        new_tids = [line.tid for line in frame]

        for tid in existing_tids:
            if tid not in new_tids:
                tracklets.append(tracks[tid])
                del tracks[tid]

        for line in frame:
            player = Player(line)

            if not player.oncourt:
                continue

            if player.tid in tracks:
                # add to track
                tracks[player.tid].append(player)
            else:
                # start new track
                tracks[player.tid] = [player]

    # Wrap up existing tracklets
    for track in tracks.values():
        tracklets.append(track)

    # Subsample
    sampled_tracklets = []
    for tracklet in tracklets:
        if len(tracklet) < args.sampling_rate:
            continue
        org_len = len(tracklet)
        sampled_tracklet = [tracklet[i] for i in range(0,len(tracklet),args.sampling_rate)]
        print(f'Downsampled tracklet {tracklet[0].tid} from {org_len} to {len(sampled_tracklet)}')
        sampled_tracklets.append(sampled_tracklet)

    return sampled_tracklets


class Player():
    def __init__(self, entry):
        self.fnum = entry.fnum
        self.tid = int(entry.tid)
        self.x = int(entry.x)
        self.y = int(entry.y)
        self.w = int(entry.w)
        self.h = int(entry.h)
        self.aspect_goal = args.crop_dims[1] / args.crop_dims[0]
        
        self.center_x = self.x + self.w//2
        self.center_y = self.y + self.h//2

        self.oncourt = (self.center_x <= court_bounds.right and
                        self.center_x >= court_bounds.left and
                        self.center_y >= court_bounds.top and
                        self.center_y <= court_bounds.bot)

    def onscreen(self, l,r,t,b):
        img_w, img_h = args.width, args.height

        if l<0:
            delta_x = abs(l)
            l += delta_x
            r += delta_x

        if r>=img_w:
            delta_x = r - (img_w -1)
            l -= delta_x
            r -= delta_x

        if t<0:
            delta_y = abs(t)
            t += delta_y
            b += delta_y

        if b>=img_h:
            delta_y = b - (img_h - 1)
            t -= delta_y
            b -= delta_y

            
        return l, r, t, b

    def compute_crop(self):
        aspect = self.h/self.w
        if aspect < self.aspect_goal:
            # need to pad height
            h = int(self.w * self.aspect_goal)
            w = self.w
        else:
            # need to pad width
            w = self.h // self.aspect_goal
            h = self.h

        l, t = self.center_x - w//2, self.center_y - h//2
        r, b = l+w, t+h

        l,r,t,b = self.onscreen(l,r,t,b)
        return l,t,r,b

    def crop(self, output_dir, idx):
        coords = self.compute_crop()

        frame_fn = os.path.join(args.img_root, f'{self.fnum:06d}.png')
        frame = Image.open(frame_fn)

        player = frame.crop(coords)
        player = player.resize(args.crop_dims)
        player_fn = f'{output_dir}/{idx:08}.png'
        player.save(player_fn)
        print(f'wrote crop {player_fn}')


def dump_tracklets(tracklets):
    """
    Write images out to a folder such that images within a tracklet are
    grouped together in number/name.
    """  
    idx = 0
    basename = osp.basename(args.img_root)
    output_dir = osp.join(args.output_root, basename)
    os.makedirs(output_dir, exist_ok=True)

    for dir_idx, tracklet in enumerate(tracklets):

        for player in tracklet:
            player.crop(output_dir, idx)
            idx += 1


def record_tracklets(tracklets):
    """
    Write images out to a folder such that images within a tracklet are
    grouped together in number/name.
    """
    crops_per_frame = defaultdict(list)
    idx = 0
    for dir_idx, tracklet in enumerate(tracklets):
        for player in tracklet:
            fnum = player.fnum
            coords = player.compute_crop()
            crops_per_frame[fnum].append((idx, coords))
            idx += 1

    return crops_per_frame


def crop_frames(crops_per_frame, cap):
    """
    Read video frame stream and take crops according to crop_per_frame,
    which is a per-frame dict of lists of crops
    """
    basename = osp.splitext(osp.basename(args.vid_fn))[0]
    output_dir = osp.join(args.output_root, basename)
    os.makedirs(output_dir, exist_ok=True)
    
    fnum = 1
    while True:
        ret_val, frame = cap.read()
        if ret_val:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = Image.fromarray(frame)
            if fnum in crops_per_frame:
                for crop in crops_per_frame[fnum]:
                    idx, coords = crop
                    player_img = frame.crop(coords)
                    player_img = player_img.resize(args.crop_dims)
                    player_fn = f'{output_dir}/{idx:08}.jpg'
                    player_img.save(player_fn)
                    print(f'wrote crop {player_fn}')
        else:
            break
        fnum += 1
        
            
            
def main():
    cap = cv2.VideoCapture(args.vid_fn)
    args.width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    args.height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    fps = cap.get(cv2.CAP_PROP_FPS)

    tracklets = build_tracklets()
    crops_per_frame = record_tracklets(tracklets)
    crop_frames(crops_per_frame, cap)


main()


"""
Methodology:
 * run FOTs on video, get results file
 * run dump_tracklets on results file
"""
