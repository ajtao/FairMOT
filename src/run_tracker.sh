
MODEL=fairmot_dla34
CONF=0.3

for DSET in Georgia_Missouri OhioState_Nebraska Northwestern_Illinois Portland_USD
do
    DATA_ROOT=/home/atao/data/vball/${DSET}
    LOGROOT=/home/atao/output/FairMOT/${DSET}
    mkdir -p $LOGROOT

    MATCH_CSV="--match_csv ${DATA_ROOT}/match_data.csv"
    OF=video   # set to 'text' for no video

    python demo.py mot --load_model /home/atao/checkpoints/FairMOT/${MODEL}.pth --conf_thres $CONF --input-video ${DATA_ROOT}/match.mp4 --output-root ${LOGROOT} --output-format $OF $MATCH_CSV

    # copy files over
    VID_FN=${DATA_ROOT}/match.mp4
    FAIRMOT_OUTPUT_DIR=/home/atao/output/FairMOT/${DSET}
    mkdir -p ${DATA_ROOT}/fairmot
    cp ${FAIRMOT_OUTPUT_DIR}/match.mp4 ${DATA_ROOT}/fairmot
    cp ${FAIRMOT_OUTPUT_DIR}/results.txt ${DATA_ROOT}/fairmot
       
    # extract single frame of each vid:
    INPUT=${DATA_ROOT}/fairmot/match.mp4
    ffmpeg -ss 00:1:00 -i $INPUT -vframes 1 -q:v 2 ${DATA_ROOT}/fairmot/match_frame.png
    
done

