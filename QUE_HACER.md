# Qué tienes que hacer tú (≈ 1 hora de trabajo tuyo + ≈ 4–5 h de GPU desatendida (1A y 1B en paralelo), perfil «urgente»)

Todo el código, la configuración, el pre-registro y el libro están listos. Las cifras salen solas de
4 notebooks en **Kaggle**, que da GPU gratis (≈ 30 h/semana) y tiene los datos de ambas competiciones
sin descargar nada. Los resultados se escriben automáticamente en el libro.

## Paso 0 · Preparación (10 min, una sola vez)

1. Crea o inicia sesión en tu cuenta de <https://www.kaggle.com> y **verifica tu teléfono**
   (*Settings → Phone verification*). Sin eso no se activan la GPU ni Internet.
2. Entra a las dos competiciones y pulsa **Join / I understand and accept** en la pestaña *Rules*:
   * <https://www.kaggle.com/competitions/sartorius-cell-instance-segmentation>
   * <https://www.kaggle.com/competitions/jarvis-calling-hiring-contest>
3. Sube este proyecto como dataset: *Create → New Dataset* → arrastra `proyecto-integrador.zip` →
   nombre **`proyecto-integrador`** → *Create* (puede ser privado).

## Paso 1 · Lanzar los 3 notebooks de cómputo (5 min; luego corren solos)

Para cada uno de `kaggle/1A_segmentacion_modelos.ipynb`, `kaggle/1B_segmentacion_ablaciones.ipynb` y
`kaggle/2_nlp.ipynb`:

1. *Create → New Notebook* → *File → Import Notebook* → elige el `.ipynb`.
2. Panel derecho → *Add Input*: el dataset **proyecto-integrador** y la **competición** que diga el
   encabezado del notebook (Sartorius para 1A y 1B, Jarvis para 2).
3. *Settings*: **Accelerator = GPU T4 x2 o P100** e **Internet = On**.
4. **Save Version → Save & Run All (Commit)**. Puedes cerrar el navegador; corre en segundo plano.

Cada notebook empieza con una **prueba de humo de ~10–15 min** que ejecuta todo en miniatura. Si algo
falla, falla ahí y no después de horas. En ese caso abre la versión → *Logs*, copia el error y
envíamelo. Lo corrijo, vuelves a subir el zip (*New Version* del dataset) y relanzas. Lo ya terminado
no se repite.

**Perfil «urgente» (por defecto):** son mis estimaciones, no medidas. Para más rigor, cambia `BUDGET` a `rapido` (≈ 10–14 h) en la 2.ª celda de cada notebook; el pre-registro ya declara la reducción en `desviaciones`.

| Notebook | Qué hace | Tiempo estimado |
|---|---|---|
| 1A | Los 6 paradigmas de segmentación × 1 semilla, test sellado, robustez, LOCO | ≈ 2.5–3.5 h (+ ≈ 40 min de instalación y prueba de humo) |
| 1B | Fracciones de datos (H1) y ablaciones mínimas, 1 semilla | ≈ 1.5–2 h (+ ≈ 15 min de instalación y humo) |
| 2 | NLP: HPO (5 trials), CV 5×3, transformer 4 épocas, test, preprocesamiento, curvas | ≈ 1.5–2.5 h (+ ≈ 30 min de humo) |

1A, 1B y 2 pueden lanzarse a la vez si Kaggle te lo permite; si no, uno tras otro. Los tiempos son
estimaciones; los reales quedan registrados.

**Revisa en la primera salida del notebook 2** qué columnas detectó como texto y etiqueta. Si no son
correctas, edita `configs/nlp.yaml` (`text_col`, `label_col`), sube una nueva versión del dataset y
relánzalo.

## Paso 2 · Notebook final (5 min tuyos + ≈ 1 h de GPU)

`kaggle/3_reporte_final.ipynb`: impórtalo igual que los anteriores. En *Add Input* agrega las
**salidas** de 1A, 1B y 2 (*Add Input → Your Work → Notebooks*), las dos competiciones y el dataset.
Activa GPU e Internet y ejecuta *Save & Run All*. Hace esto:

* entrena y registra los modelos desplegables;
* corre las pruebas;
* ejecuta los notebooks 00–07 con los datos reales, incluido el experimento de deriva;
* **rellena el libro automáticamente**: tablas, veredictos de H1–H6, model cards y datasheets;
* construye el Jupyter Book.

Al final descarga **`ENTREGA.zip`** de la pestaña *Output*. Ese es el entregable.

## Paso 3 · Lo único que debes escribir o revisar tú (≈ 30–45 min)

1. **`book/conclusiones.md`**: una frase en «Resultado más inesperado» (mira `book/_generado/hipotesis.md`).
2. **`book/anexos/declaracion_ia.md`**: tu nombre, la fecha y quién revisó qué. Es obligatoria y debe
   ser honesta.
3. **`book/_config.yml`**: pon tu nombre en `author`.
4. **Revisiones manuales (opcionales, suman puntos):**
   * 20–50 imágenes: abre `reports/tables/seg_revision_manual_PLANTILLA.csv`, cuenta células sin anotar o
     mal anotadas y guárdalo como `seg_revision_manual.csv`.
   * Errores de texto: `reports/tables/nlp_errores_para_anotar.csv` con dos anotadores (tú y un
     compañero) para el κ de Cohen.

   Si no las haces, el libro ya lo declara como limitación.
5. Lee el libro (`book/_build/html/index.html`) de principio a fin, unos 30 min, para la defensa.

## Paso 4 · Defensa oral (15 min)

Cuando tengas `ENTREGA.zip` puedo prepararte las diapositivas con tus resultados reales. Los temas son:
decisiones de diseño, amenazas a la validez y resultados negativos. El guion de preguntas difíciles está
en `book/anexos/revision_por_pares.md`.

## Si te quedas sin tiempo

Con solo **1A + 2 + 3** ya tienes la comparación de todos los modelos, H2–H6 y el despliegue; sin 1B
faltan las ablaciones y H1, y el libro las marca como «pendiente». Con `BUDGET=completo` en la primera
celda se usa el presupuesto completo (más HPO y épocas), si tienes más días.
