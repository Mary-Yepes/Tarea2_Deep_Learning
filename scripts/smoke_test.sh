#!/usr/bin/env bash
# Prueba de humo: levanta el contenedor, espera /ready y llama a /predict_text y /predict_image.
set -euo pipefail
IMAGE="${1:-proyecto-integrador:latest}"
CID=$(docker run -d -p 8000:8000 "$IMAGE")
trap 'docker logs "$CID" | tail -20; docker rm -f "$CID" >/dev/null' EXIT
for i in $(seq 1 30); do
  curl -sf localhost:8000/ready >/dev/null && break || sleep 2
done
curl -sf localhost:8000/health
curl -sf -X POST localhost:8000/predict_text -H 'content-type: application/json' \
     -d '{"text":"python sql pipeline deploy"}' | tee /dev/stderr | python -c "import json,sys; b=json.load(sys.stdin); assert b['label'] and abs(sum(b['probabilities'].values())-1)<1e-6"
python - <<'PY'
import sys; sys.path.insert(0, "src")
import cv2
from project.data_seg import make_synthetic_image
img, _ = make_synthetic_image((128, 160), "cort", seed=0)
cv2.imwrite("/tmp/smoke.png", img)
PY
curl -sf -X POST localhost:8000/predict_image -F "file=@/tmp/smoke.png" | python -c "import json,sys; b=json.load(sys.stdin); assert b['n_instances']==len(b['rles']); print('n_instances', b['n_instances'])"
curl -sf localhost:8000/model_info >/dev/null
echo "PRUEBA DE HUMO OK"
