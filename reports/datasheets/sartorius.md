# Datasheet — Sartorius Cell Instance Segmentation

Formato: Gebru et al. (2021), *Datasheets for Datasets*. Las cifras marcadas con **[verificar]** provienen
de la documentación pública de la competición o de la literatura. Se contrastan con los archivos reales
en `notebooks/01_seg_eda_audit.ipynb` (sección 1 y pruebas de integridad) y toda discrepancia se anota
en la tabla final.

## Motivación
* **Propósito.** Es una competición de Kaggle (Sartorius, 2021) para segmentar instancias de células
  neuronales en microscopía de contraste de fase, con el fin de medir el efecto de tratamientos en
  investigación de enfermedades neurodegenerativas.
* **Creadores y financiación.** Sartorius AG, con imágenes en la línea de LIVECell (Edlund et al., 2021).

## Composición
| Elemento | Esperado | Real (EDA) |
|---|---|---|
| Imágenes anotadas (train) | 606 PNG de 520 × 704 px, 8 bits, un canal **[verificar]** | *[completar]* |
| Tipos celulares | `shsy5y` (≈ 155 imágenes), `astro` (≈ 131), `cort` (≈ 320) **[verificar]** | *[completar]* |
| Instancias (filas de `train.csv`) | ≈ 73 600 **[verificar]** | *[completar]* |
| Columnas | `id, annotation, width, height, cell_type, plate_time, sample_date, sample_id, elapsed_timedelta` | *[completar]* |
| Codificación de máscaras | RLE 1-indexado, aplanado por filas (orden C) | Se verifica visualmente (cap. 1 del notebook 01) |
| Sin anotar | `train_semi_supervised/` (≈ 1 970 imágenes) **[verificar]** | *[completar]* |
| Datos externos | `LIVECell_dataset_2021/` (formato COCO) | *[completar]* |
| Test | Oculto (Kaggle); **no se usa**: el test del proyecto es un 15 % sellado de train | — |

* **Unidad y metadatos.** Cada fila es una instancia. Los metadatos de placa y muestra (`sample_id`,
  `sample_date`, `plate_time`, `elapsed_timedelta`) definen **grupos experimentales**: las particiones
  se agrupan por `sample_id`.
* **Subgrupos relevantes.** Tipo celular, densidad (células por imagen y cobertura) y tamaño (área y
  diámetro equivalente).
* **Solapes.** Existen máscaras solapadas en la referencia **[verificar]**. Al construir el mapa de
  etiquetas prevalece la primera instancia, y el número de píxeles en conflicto se reporta por imagen.

## Proceso de recolección y anotación
* Las imágenes provienen de un sistema de imagen de células vivas (contraste de fase) y se anotaron por
  instancia, de forma manual o asistida **[verificar]**.
* **Calidad de anotación conocida.** La comunidad reportó anotaciones incompletas en `astro` (células
  sin anotar y máscaras rotas o fragmentadas) y neuritas finas difíciles de delimitar en `shsy5y`
  **[verificar]**. La cota de ruido se estima revisando a mano 50 imágenes
  (`reports/tables/seg_revision_manual.csv`).

## Usos
* **Previstos.** Investigación y docencia en segmentación de instancias, y comparación metodológica.
* **No previstos.** Diagnóstico clínico, decisiones sobre pacientes y despliegue en otros microscopios o
  protocolos sin revalidar (ver la robustez en el notebook 03).

## Distribución, licencia y términos de uso
* Reglas de la competición de Kaggle: uso no comercial y académico; no redistribuir los datos.
  **[verificar en la pestaña *Rules*]**. LIVECell: CC BY-NC 4.0 **[verificar]**.
* Los datos **no** se incluyen en este repositorio ni en la imagen Docker. Se versionan localmente con
  DVC (`data/raw/*.dvc` contiene solo hashes).

## Riesgos de privacidad
No hay datos personales: son imágenes de cultivos celulares. Los metadatos de fecha y placa no
identifican personas.

## Limitaciones conocidas
* Hay solo tres tipos celulares, de un único laboratorio, microscopio y protocolo, lo que limita la
  validez externa.
* El conteo y la densidad son muy desiguales entre tipos, así que la métrica global depende de la
  composición del test.
* Las anotaciones de los tipos con morfología compleja son ruidosas, lo que acota el AP alcanzable.

## Discrepancias con el enunciado (registro)
| Fecha | Supuesto del enunciado | Observado en los archivos | Acción |
|---|---|---|---|
| *[completar al ejecutar el notebook 01]* | | | |
