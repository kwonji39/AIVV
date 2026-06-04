#For semi conductor equipment from Birck
cd /Users/ujinjeon/python/Anomaly_detection
/Users/ujinjeon/python/Anomaly_detection/.venv/bin/python Anomaly_detection_F200.py --epochs 100 --window-size 50 --mc-samples 50 --plot

#For UUV (Remus 100 simulink)

cd /Users/ujinjeon/python/Anomaly_detection && /Users/ujinjeon/python/Anomaly_detection/.venv/bin/python Anomaly_detection_F200.py \
  --dataset uuv_yaw \
  --uuv-file UUV_yaw_fault.txt \
  --uuv-downsample 20 \
  --uuv-train-frac 0.7 \
  --epochs 100 \
  --window-size 50 \
  --mc-samples 50 \
  --heteroscedastic \
  --forecast-horizon 20 \
  --fault-persistence 3 \
  --plot