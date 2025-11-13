Proyecto de Predicción Visual de la Evolución de Lesiones Cutáneas (GAN)

📋 Descripción del Proyecto

Este repositorio contiene el script/notebook de preprocesamiento para el proyecto de predicción visual de la evolución de lesiones cutáneas. El objetivo de este preprocesamiento no es clasificar, sino unificar los datasets HAM10000 e ISIC (2018/2019) para extraer pares de imágenes longitudinales (T_0, T_1).

El objetivo final es preparar un conjunto de datos en el formato requerido por una arquitectura GAN de traducción imagen-a-imagen (por ejemplo, Pix2Pix). El modelo entrenará para predecir la imagen en tiempo T_1 (salida) a partir de la imagen en tiempo T_0 (entrada).

🗂️ Datasets fuente

- HAM10000
  - Ruta: ./Datos/HAM10000/
  - Contenido relevante:
    - HAM10000_images_part_1/
    - HAM10000_images_part_2/
    - HAM10000_metadata.csv

- ISIC 2018 / ISIC 2019
  - Ruta: ./Datos/ISIC2018/
  - Contenido relevante:
    - ISIC2018_Task3/ (imágenes)
    - ISIC2018_Task3_Training_LesionGroupings.csv

🔧 Proceso de preprocesamiento (resumen)

El notebook `Preentrenamiento_v1.0.ipynb` realiza los pasos principales:

1. Indexado de imágenes: escanea las carpetas de imágenes (HAM10000 e ISIC) y crea un mapa `image_id -> ruta_archivo`.
2. Lectura de metadatos (p. ej. `HAM10000_metadata.csv`).
3. Agrupado por `lesion_id` para encontrar lesiones con más de una imagen (datos longitudinales).
4. Creación de pares (T_0, T_1): se toma la primera imagen como `train_A` (entrada) y la segunda como `train_B` (objetivo).
5. Copia y renombrado: los pares se copian a la estructura de salida en `Preprocesado_Pix2Pix/` y se renombran para que el mismo nombre de archivo exista en `train_A/` y en `train_B/`.

> Nota: el mismo nombre de archivo en `train_A` y `train_B` permite emparejarlos fácilmente por el data loader de Pix2Pix.

📁 Estructura de salida esperada (formato Pix2Pix)

Training/
├── Datos/
│   ├── HAM10000/
│   ├── ISIC2018/
│   └── Preprocesado_Pix2Pix/
│       ├── train_A/   # imágenes T_0
│       └── train_B/   # imágenes T_1

Ejemplo de par: `train_A/HAM_0000118_ISIC_0027419.jpg` y `train_B/HAM_0000118_ISIC_0027419.jpg`.

💻 Instalación y uso

Requisitos mínimos (ejemplo):

```bash
pip install pandas tqdm
```

Ejecución:
- Abrir `Preentrenamiento_v1.0.ipynb` y ejecutar las celdas en orden. El notebook detectará las rutas en `Datos/` y generará `Preprocesado_Pix2Pix/` con los pares.

🔎 Comando para descomprimir `dataset_unificado.zip` (Windows, desde cmd.exe)

Usando PowerShell desde cmd.exe (descomprime en carpeta `dataset_unificado`):

```cmd
powershell -Command "Expand-Archive -LiteralPath 'dataset_unificado.zip' -DestinationPath 'dataset_unificado' -Force"
```

Alternativa con 7-Zip (si está instalado):

```cmd
"C:\Program Files\7-Zip\7z.exe" x dataset_unificado.zip -odataset_unificado
```

🛠️ Tecnologías

- Python 3.x
- pandas
- tqdm
- shutil, os
- Jupyter Notebook

📌 Notas importantes

- Asegúrate de que `Datos/HAM10000/` y `Datos/ISIC2018/` existan y contengan los archivos esperados antes de ejecutar el preprocesamiento.
- La clase `SCC` fue omitida intencionalmente en el proceso actual.
- Los pares están preparados para Pix2Pix: mismo nombre en `train_A/` y `train_B/`.

📅 Fecha de última actualización: 28 de octubre de 2025
🔖 Versión: 1.0

Pasos para ejecutar EntrenamientoPreliminar1_1:
en la consola ejecutar este comando:
pip install -r requirements.txt
luego ejecutar el notebook Preentrenamiento_v1.0.ipynb