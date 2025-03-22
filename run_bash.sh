DATASET_DIR=$1
PARTICIPANT=$2
STATIC_MODULE=$3
DYNAMIC_MODULE=$4
DYNAMIC_NUM=$5

# DATASET_DIR="D:/Data/pface"
# PARTICIPANT="FaceData"
# STATIC_MODULE="5_Static"
# DYNAMIC_MODULE="6_Dynamic"
# DYNAMIC_NUM="1"

echo "================================================================"
if [ -z "$PARTICIPANT" ]
then
    echo "[E:] PARTICIPANT is empty"
    exit
else
    echo "PARTICIPANT = $PARTICIPANT"
fi
if [ -z "$STATIC_MODULE" ]
then
    echo "[E:] STATIC_MODULE is empty"
    exit
else
    echo "STATIC_MODULE = $STATIC_MODULE"
fi
if [ -z "$DYNAMIC_MODULE" ]
then
    echo "[E:] DYNAMIC_MODULE is empty"
    exit
else
    echo "DYNAMIC_MODULE = $DYNAMIC_MODULE"
fi
if [ -z "$DYNAMIC_NUM" ]
then
    echo "[E:] DYNAMIC_NUM is empty"
    exit
else
    echo "DYNAMIC_NUM = $DYNAMIC_NUM"
fi
echo "================================================================"

# running static pbrdf
echo "---------------- static pbrdf ----------------"
python ./pbrdf_python/optimize.py -config_path ./pbrdf_python/config/pface_large.json -root_dir_path $DATASET_DIR -participants_name $PARTICIPANT -module_name $STATIC_MODULE -num_frames 0;
python ./pbrdf_python/test.py -config_path ./pbrdf_python/config/test_pface.json -root_dir_path $DATASET_DIR -participants_name $PARTICIPANT -module_name $STATIC_MODULE -num_frames 0;

for NUM in $DYNAMIC_NUM;
do
    # running dynamic pbrdf
    echo "---------------- dynamic pbrdf ${DYNAMIC_MODULE}_${NUM} ----------------"
    python ./pbrdf_python/optimize_dynamic.py -config_path ./pbrdf_python/config/pface_dynamic_large.json -root_dir_path $DATASET_DIR -participants_name $PARTICIPANT -module_name "$DYNAMIC_MODULE"_"$NUM" -num_frames 0 -static_module_name $STATIC_MODULE;
    python ./pbrdf_python/test_dynamic.py -config_path ./pbrdf_python/config/test_pface_dynamic_large.json -root_dir_path $DATASET_DIR -participants_name $PARTICIPANT -module_name "$DYNAMIC_MODULE"_"$NUM" -num_frames 0;
done
