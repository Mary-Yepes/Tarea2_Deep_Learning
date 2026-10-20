# Modelos reutilizados: entrenamiento con las MISMAS particiones

Todos los modelos externos se entrenan y ajustan con `data/processed/seg_splits.csv`. Las particiones se
exportan con `python scripts/seg_experiments.py export --format coco cellpose hovernet`, y la evaluación
se hace siempre con `scripts/seg_experiments.py evaluate`, que usa el mismo código de métricas para todos.
En la *model card* de cada modelo se registran la versión, el commit, la licencia y las modificaciones.

| Modelo | Repositorio / versión | Licencia | Entrenamiento (resumen) | Adaptador |
|---|---|---|---|---|
| Mask R-CNN + Swin-T | open-mmlab/mmdetection v3.x, config `mask-rcnn_swin-t-p4-w7_fpn_1x_coco` | Apache-2.0 | `tools/train.py configs/mmdet/mask-rcnn_swin-t_cells.py` con `data/processed/coco/annotations_{train,val}.json`, 1 clase, anchors del EDA, `max_per_img` ≈ 1.2 × p99 del conteo | `maskrcnn_mmdet:<cfg>,<ckpt>` |
| Mask R-CNN (alternativa) | torchvision `maskrcnn_resnet50_fpn_v2` | BSD-3 | Si se usa, es una desviación declarada (backbone ResNet en lugar de Swin) | `maskrcnn_tv:<pesos>` |
| Hover-Net | vqdang/hover_net (commit fijado en la model card) | MIT | `config.py`: `nr_type=None` (solo ramas NP + HV); parches de `data/processed/hovernet/`; pérdidas BCE+Dice y MSE+MSGE | `hovernet:<modulo:funcion>` (la función devuelve `np_prob, hv`) |
| Cellpose 2.0 | `cellpose>=3,<4` | BSD-3 | Generalista: `cyto2` sin ajuste. Ajustado: `python -m cellpose --train --dir data/processed/cellpose/train --test_dir data/processed/cellpose/val --pretrained_model cyto2 --chan 0 --n_epochs 500 --learning_rate 0.1 --mask_filter _masks`. Humano en el bucle: corregir 20 imágenes de validación en la GUI y re-entrenar | `cellpose:cyto2` / `cellpose:<ruta>` |
| SAM / MedSAM | facebookresearch/segment-anything; bowang-lab/MedSAM | Apache-2.0 | (a) cero-shot; (b) prompts de caja desde Mask R-CNN; (c) ajuste fino del decodificador o LoRA (peft) con las máscaras de train | `sam_auto:<ckpt>` / `SAMBoxAdapter` |

**Presupuesto equitativo.** 30 trials de HPO y 6 horas-GPU por modelo, según `configs/prereg.yaml`. Para
los modelos pesados, la HPO se hace sobre un único split de validación, seguida de 5-fold con
configuración fija. El sesgo optimista resultante se declara en el reporte.
