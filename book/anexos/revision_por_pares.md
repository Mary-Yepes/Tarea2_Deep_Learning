# Anexo B · Autoevaluación crítica tipo revisión por pares

Estas son las cinco objeciones más serias que haría un revisor externo, cada una con su respuesta: la
evidencia que la atiende o el reconocimiento explícito de la limitación.

**Objeción 1. «La comparación entre paradigmas no es justa: los modelos preentrenados traen millones de
imágenes de cómputo previo que el presupuesto de 6 horas-GPU no contabiliza.»**
*Respuesta.* Es cierto y se reconoce. El presupuesto equitativo pre-registrado iguala el cómputo **de
ajuste** (30 trials y horas-GPU), no el costo total de crear el modelo. Por eso la pregunta se formula en
términos de uso (H1: ¿conviene un preentrenado cuando hay pocos datos?) y se complementa con la ablación
de preentrenamiento sobre la **misma** arquitectura propia (U-Net++ desde cero vs ImageNet vs
LIVECell), que aísla el efecto del preentrenamiento sin cambiar la arquitectura.

**Objeción 2. «Con unas 600 imágenes y ~3 placas por tipo en test, las conclusiones por tipo celular
tienen muy poca potencia, y el bootstrap por imagen subestima la incertidumbre.»**
*Respuesta.* Por eso el IC es **jerárquico** (remuestrea placas y luego imágenes), y la verificación con
el modelo mixto (intercepto aleatorio por `sample_id`) es obligatoria. La prueba
`test_hierarchical_bootstrap_wider_than_naive_with_clusters` muestra que el IC jerárquico es más ancho que
el ingenuo cuando hay correlación intra-placa. Se reconoce que, en estratos pequeños, «no significativo»
no equivale a «equivalente»: solo se declara equivalencia cuando el IC queda dentro de ±δ.

**Objeción 3. «AP@[0.5:0.95] mide la coincidencia con anotaciones ruidosas, sobre todo en astrocitos;
un modelo que segmenta mejor que el anotador sale penalizado.»**
*Respuesta.* Es una amenaza a la validez de constructo y se declara. Se estima una cota de ruido de
anotación con revisión manual de 50 imágenes y se reportan las métricas estratificadas por tipo. En la
taxonomía de errores, las «falsas» detecciones en astro se inspeccionan a mano para separar las falsas
reales de las células visibles sin anotar. Con más presupuesto se haría una doble anotación de un
subconjunto de test para estimar el techo.

**Objeción 4. «En NLP, la etiqueta y su proceso de generación son desconocidos; además, los proxies de
género inferidos del texto son demasiado débiles para hablar de equidad.»**
*Respuesta.* Se acepta. La *datasheet* registra el origen de la etiqueta como desconocido si la
competición no lo documenta. El análisis de equidad se limita a lo defendible: pruebas
**contrafactuales** (intercambio de términos de género y edad, inserción controlada), que no requieren
atributos, y métricas por subgrupo **solo** si hay ≥ 30 ejemplos por grupo con un proxy explícito. La
conclusión normativa (no usar el modelo para decidir automáticamente) **no** depende de que esas
métricas salgan bien: se apoya en el marco regulatorio y en la imposibilidad de auditar subgrupos reales.

**Objeción 5. «El experimento de deriva usa perturbaciones sintéticas que el detector, por construcción,
ve con facilidad; eso no demuestra que detecte la deriva real.»**
*Respuesta.* Es parcialmente cierto. Por eso el diseño incluye derivas que **no** cambian la
distribución marginal de las características de entrada: cambio de proporciones de clase (tema/prior) y
de tipo celular. La evidencia de la versión demo muestra que esas derivas son más difíciles y se detectan
sobre todo por la deriva de predicciones. Los umbrales por defecto de PSI no controlan la tasa de falsas
alarmas con ventanas pequeñas: se demuestra que E[PSI] ≈ (B − 1)(1/n + 1/m) bajo la nula y por eso los
umbrales se calibran por simulación. Se reconoce que la validación definitiva requiere un periodo o una
fuente realmente distintos, y LIVECell sirve como deriva real de protocolo.
