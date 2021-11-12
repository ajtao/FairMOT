
DATA_ROOT=/home/atao/data/vball
VID=womens_clip_2plays
MODEL=fairmot_dla34
CONF=0.3

LOGROOT=/home/atao/output/FairMOT/${VID}_${MODEL}_${CONF}

OF=video  # set to text for no video

MATCH_CSV="--match_csv ${DATA_ROOT}/${VID}_data.csv"
MATCH_CSV=""
MAX_SECONDS=""

python demo.py mot --load_model ../models/${MODEL}.pth --conf_thres $CONF --input-video ${DATA_ROOT}/${VID}.mp4 --output-root ${LOGROOT} --output-format $OF $MATCH_CSV $MAX_SECONDS
