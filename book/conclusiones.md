# Conclusiones, limitaciones y trabajo futuro

## Qué se pondría en producción y bajo qué supuestos

* **Imágenes.** El modelo con mejor puntaje multicriterio *robusto*, es decir, que no cambie al
  perturbar los pesos. Su decisión **deja de ser válida** si (i) cambia el microscopio, el aumento o el
  protocolo de cultivo (el monitor detecta deriva en intensidad, contraste, enfoque o número de
  instancias), (ii) aparece un tipo celular no visto (caída LOCO > δ) o (iii) la densidad supera el
  rango observado (las fusiones crecen con la densidad).
* **Texto.** Solo como **apoyo con supervisión humana**. Si H4 se sostiene (equivalencia práctica), se
  prefiere el modelo clásico calibrado por costo, latencia y explicabilidad. La decisión deja de ser
  válida ante deriva de vocabulario o de temas, cambios en la proporción de clases o evidencia de
  disparidad entre subgrupos.

**Con las cifras de esta corrida (perfil «urgente», preliminares).** En imágenes, el candidato es Cellpose
ajustado (AP 0.236, primero en los tres tipos celulares), con la condición de reajustarlo si cambia el tipo
celular o el desenfoque. En texto, TF-IDF + XGBoost es la opción razonable: DistilBERT lo iguala en F1
(0.712 frente a 0.708) sin una ventaja demostrada (H4 no concluyente) y cuesta unas 5.7 veces más por
texto. No se evaluó la equidad por subgrupos con datos suficientes, por lo que el uso en selección de
personal solo sería de apoyo con supervisión humana.

## Hipótesis

Veredictos calculados automáticamente con los criterios pre-registrados (`scripts/fill_report.py`):

```{include} _generado/hipotesis.md
```

**Resultado más inesperado:** la curva de aprendizaje de U-Net++ no fue gradual. Con el 25 % de las
imágenes de entrenamiento el AP fue ≈ 0,001 y con el 100 %, ≈ 0,16 (`seg_fractions.csv`). Lo esperado era
una mejora suave. Es probable que sea un efecto del presupuesto corto (12 épocas como máximo) y no de la
cantidad de datos: con pocas imágenes no se alcanzó a converger, y en el entrenamiento con todas las imágenes la pérdida de
validación seguía bajando en la última época. Por eso no se interpreta como un efecto de los datos, sino como una limitación del perfil
«urgente». En NLP hubo otra sorpresa: TF-IDF + XGBoost y la CNN-1D superaron a TF-IDF + LR en la búsqueda de
hiperparámetros, lo contrario de lo que se espera de un clasificador lineal sobre texto. Se verificó con
una sola semilla, así que queda como hipótesis y no como conclusión.

## Principales amenazas a la validez y experimentos que las reducirían

1. **Validez externa** (un solo laboratorio). Se reduciría evaluando sin reentrenar sobre LIVECell
   (otros tipos celulares y otro protocolo) y sobre un corpus de reclutamiento de otro origen.
2. **Ruido de anotación.** Se reduciría con una doble anotación de un subconjunto de test para estimar
   el techo (acuerdo entre anotadores) y reportar la métrica relativa a ese techo.
3. **Sesgo optimista en modelos pesados.** Se reduciría con validación anidada también en ellos, lo que
   multiplica el costo por ≈ 5.

## Con el doble y con la mitad del presupuesto

* **El doble.** Anidada completa para los modelos pesados, 5 semillas en todos, pre-entrenamiento
  auto-supervisado con las ~2 000 imágenes sin anotar y ensamble de instancias.
* **La mitad.** Se priorizan el baseline clásico, U-Net++ propia y Cellpose ajustado en imágenes, y
  TF-IDF + LR/XGBoost y un solo transformer en NLP, con 3 semillas. Se elimina SAM-LoRA, que es el de
  mayor costo y el que tiene la brecha de dominio más incierta.

## Reflexión ética y de impacto

* **Usos indebidos.** Filtrado automático de candidatos sin supervisión; uso de proxies textuales
  (nombres, instituciones, lugares) que reproducen discriminación histórica; extrapolación de conteos
  celulares a conclusiones clínicas.
* **Poblaciones afectadas.** Candidatos con trayectorias o redacción atípicas (otro idioma nativo,
  plantillas distintas), que son los más afectados por los atajos léxicos.
* **Salvaguardas.** Supervisión humana obligatoria; derivación por incertidumbre (conforme y selectiva);
  auditoría por subgrupos con datos auto-declarados; *model cards* públicas; monitoreo de deriva con
  *rollback*; y vía de impugnación para las personas afectadas.
