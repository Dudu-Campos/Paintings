#!/usr/bin/env bash
# Places-LT: precisa dos tars do Places365-Standard (24 GB + 0,5 GB) para
# extrair só as 106.300 imagens listadas nos splits LT. wget -c retoma.
set -u
cd /exp/duducampos/Paintings
for attempt in $(seq 1 40); do
  echo "=== [$(date +%H:%M:%S)] tentativa $attempt" | tee -a logs/places_lt_download.log
  wget -c -O data/places_lt/tars/val_256.tar \
    "http://data.csail.mit.edu/places/places365/val_256.tar" >> logs/places_lt_download.log 2>&1
  wget -c -O data/places_lt/tars/train_256.tar \
    "http://data.csail.mit.edu/places/places365/train_256_places365standard.tar" >> logs/places_lt_download.log 2>&1
  status=$?
  size=$(stat -c%s data/places_lt/tars/train_256.tar 2>/dev/null || echo 0)
  echo "=== [$(date +%H:%M:%S)] codigo $status, train_256=$((size/1024/1024)) MB" | tee -a logs/places_lt_download.log
  [ "$size" -ge 26103685120 ] && { echo "=== completo" | tee -a logs/places_lt_download.log; break; }
  sleep 15
done
