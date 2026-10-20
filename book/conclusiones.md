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

## Hipótesis

Veredictos calculados automáticamente con los criterios pre-registrados (`scripts/fill_report.py`):

```{include} _generado/hipotesis.md
```

**Resultado más inesperado:** *[completar]*.

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
