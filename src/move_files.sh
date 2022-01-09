# Dump out player crops that were identified by tracker/detector

DATASET=(still have to move :
  -rwxrwxrwx 1 atao atao 5187167542 Jan  7 20:41 PSU_Towson-002.mp4
  -rwxrwxrwx 1 atao atao 2063789024 Jan  7 20:38 UCF_Pepp.zip
  -rwxrwxrwx 1 atao atao 1942405207 Jan  7 20:38 Stanford_USC.zip
  -rwxrwxrwx 1 atao atao 1799379548 Jan  7 20:37 Utah_Colorado.zip
  -rwxrwxrwx 1 atao atao 1406988293 Jan  7 20:35 Washington_Hawaii.zip
  -rwxrwxrwx 1 atao atao     105404 Jan  7 20:32 PSU_Towson.zip
  -rwxrwxrwx 1 atao atao 1539951007 Jan  7 20:30 SDSU_UNO.zip
  -rwxrwxrwx 1 atao atao 1277944303 Jan  7 20:29 WKU_South Carolina.zip	 

for DSET in ${DATASET[@]}
do
    DATA_DIR=/home/atao/data/vball/${DSET}
    VID_FN=${DATA_DIR}/match.mp4
    FAIRMOT_OUTPUT_DIR=/home/atao/output/FairMOT/${DSET}

    echo "*********************************"
    echo "Working on $DSET"
    echo "*********************************"

    if true; then
	# move FairMOT results over
	mkdir -p ${DATA_DIR}/fairmot
	cp ${FAIRMOT_OUTPUT_DIR}/match.mp4 ${DATA_DIR}/fairmot
	cp ${FAIRMOT_OUTPUT_DIR}/results.txt ${DATA_DIR}/fairmot
	
	# extract single frame of each vid:
	INPUT=${DATA_DIR}/fairmot/match.mp4
	ffmpeg -ss 00:1:00 -i $INPUT -vframes 1 -q:v 2 ${DATA_DIR}/fairmot/match_frame.png
    fi

    #mkdir -p $OUTPUT_DIR
    #python process_tracks.py --action dump_crops --tracking_csv $TRACK_CSV --vid_fn $VID_FN --output_dir $OUTPUT_DIR --sampling_rate $SAMPLING_RATE
done
