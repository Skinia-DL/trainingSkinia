# Proyecto de Detección de Cáncer de Piel - Dataset Unificado

## 📋 Descripción del Proyecto

Este proyecto implementa un sistema de preprocesamiento para unificar dos datasets importantes de detección de cáncer de piel: **HAM10000** e **ISIC 2019**. El objetivo es crear un dataset consolidado que combine ambas fuentes de datos para mejorar la capacidad de entrenamiento de modelos de deep learning en la clasificación de lesiones dermatológicas.

## 🗂️ Estructura del Dataset Unificado

El dataset unificado contiene las siguientes clases de lesiones dermatológicas:

| Clase | Nombre Completo | Descripción |
|-------|-----------------|-------------|
| `akiec` | Actinic Keratoses and Intraepithelial Carcinoma | Queratosis actínicas y carcinoma intraepitelial |
| `bcc` | Basal Cell Carcinoma | Carcinoma basocelular |
| `bkl` | Benign Keratosis-like Lesions | Lesiones benignas tipo queratosis |
| `df` | Dermatofibroma | Dermatofibroma |
| `mel` | Melanoma | Melanoma |
| `nv` | Melanocytic Nevi | Nevos melanocíticos |
| `vasc` | Vascular Lesions | Lesiones vasculares |

## 🔧 Proceso de Unificación

### Datasets Fuente

1. **HAM10000** (Human Against Machine with 10000 training images)
   - Ubicación: `./Datos/archive (2)/`
   - Formato: Imágenes en dos carpetas (`HAM10000_images_part_1` y `HAM10000_images_part_2`) con metadatos en CSV
   - Metadatos: `HAM10000_metadata.csv`

2. **ISIC 2019** (International Skin Imaging Collaboration 2019)
   - Ubicación: `./Datos/archive/`
   - Formato: Imágenes organizadas por carpetas de clase (AK, BCC, BKL, DF, MEL, NV, SCC, VASC)
   - Metadatos: `ISIC_2019_Training_GroundTruth.csv` y `ISIC_2019_Training_Metadata.csv`

### Mapeo de Clases

El sistema realiza un mapeo inteligente para unificar las nomenclaturas:

```python
# Mapeo ISIC 2019 -> Dataset Unificado
'AK' → 'akiec'    # Actinic Keratoses
'BCC' → 'bcc'     # Basal Cell Carcinoma
'BKL' → 'bkl'     # Benign Keratosis-like
'DF' → 'df'       # Dermatofibroma
'MEL' → 'mel'     # Melanoma
'NV' → 'nv'       # Nevus
'VASC' → 'vasc'   # Vascular Lesions
'SCC' → [EXCLUIDA] # Squamous Cell Carcinoma (no incluida en HAM10000)
```

## 🚀 Funcionalidades Implementadas

### 1. Procesamiento HAM10000 (`process_ham10000()`)
- ✅ Lectura de metadatos desde CSV
- ✅ Búsqueda eficiente de imágenes en ambas carpetas
- ✅ Indexación automática para optimizar el rendimiento
- ✅ Copia organizizada por clase de lesión

### 2. Procesamiento ISIC 2019 (`process_isic2019()`)
- ✅ Recorrido automático de carpetas de clase
- ✅ Mapeo de nomenclaturas (mayúsculas → minúsculas)
- ✅ Exclusión de la clase SCC para mantener consistencia
- ✅ Contador de imágenes procesadas por clase

### 3. Verificación del Dataset (`verify_unification()`)
- ✅ Conteo automático de imágenes por clase
- ✅ Reporte detallado del dataset unificado
- ✅ Validación de la integridad del proceso

## 💻 Instalación y Uso

### Prerrequisitos
```bash
pip install pandas
pip install tqdm
```

### Ejecución
```python
# Ejecutar el notebook Preentrenamiento_v1.0.ipynb
# O ejecutar las funciones directamente:

# Crear directorio de salida
OUTPUT_DATASET_PATH.mkdir(parents=True, exist_ok=True)

# Procesar ambos datasets
process_ham10000()
process_isic2019()

# Verificar resultado
verify_unification()
```

## 📁 Estructura de Directorios

```
Training/
├── Datos/
│   ├── archive/                    # Dataset ISIC 2019
│   │   ├── AK/                     # Actinic Keratoses
│   │   ├── BCC/                    # Basal Cell Carcinoma
│   │   ├── BKL/                    # Benign Keratosis-like
│   │   ├── DF/                     # Dermatofibroma
│   │   ├── MEL/                    # Melanoma
│   │   ├── NV/                     # Nevus
│   │   ├── SCC/                    # Squamous Cell Carcinoma (excluido)
│   │   ├── VASC/                   # Vascular Lesions
│   │   └── *.csv                   # Metadatos ISIC 2019
│   ├── archive (2)/                # Dataset HAM10000
│   │   ├── HAM10000_images_part_1/ # Imágenes parte 1
│   │   ├── HAM10000_images_part_2/ # Imágenes parte 2
│   │   └── HAM10000_metadata.csv   # Metadatos HAM10000
│   └── dataset_unificado/          # Dataset final unificado
│       ├── akiec/                  # Queratosis actínicas
│       ├── bcc/                    # Carcinoma basocelular
│       ├── bkl/                    # Lesiones benignas
│       ├── df/                     # Dermatofibroma
│       ├── mel/                    # Melanoma
│       ├── nv/                     # Nevos melanocíticos
│       └── vasc/                   # Lesiones vasculares
├── Preentrenamiento_v1.0.ipynb     # Notebook principal
└── README.md                       # Este archivo
```

## 🔍 Resolución de Problemas

### Problema Identificado y Resuelto

**Issue Original**: La función `process_isic2019()` no estaba reconociendo las carpetas del dataset ISIC 2019.

**Causa**: Discrepancia entre los nombres de carpetas (en mayúsculas: AK, BCC, MEL...) y las claves del diccionario de mapeo (en minúsculas: ak, bcc, mel...).

**Solución Implementada**:
- ✅ Actualización del diccionario `label_mapping` para usar claves en mayúsculas
- ✅ Simplificación de la lógica de exclusión de la clase SCC
- ✅ Adición de contadores informativos por clase
- ✅ Mejora en los mensajes de depuración

## 📊 Características del Dataset Final

El dataset unificado resultante combina:
- **Imágenes HAM10000**: ~10,015 imágenes de alta calidad
- **Imágenes ISIC 2019**: ~25,331 imágenes adicionales (excluyendo SCC)
- **Total Estimado**: ~35,000+ imágenes dermatológicas
- **Clases**: 7 tipos de lesiones dermatológicas
- **Formato**: Archivos JPEG organizados por carpetas de clase

## 🎯 Próximos Pasos

1. **Análisis Exploratorio**: Implementar visualizaciones del dataset unificado
2. **Balanceado de Clases**: Evaluar y corregir desbalances entre clases
3. **Preprocesamiento**: Aplicar técnicas de normalización y aumento de datos
4. **Entrenamiento**: Desarrollar modelo de clasificación con transfer learning
5. **Evaluación**: Implementar métricas de rendimiento y validación cruzada

## 🛠️ Tecnologías Utilizadas

- **Python 3.13.7
- **pandas**: Manipulación de metadatos
- **pathlib**: Manejo de rutas multiplataforma
- **shutil**: Operaciones de archivos
- **tqdm**: Barras de progreso
- **Jupyter Notebooks**: Entorno de desarrollo

## 👥 Contribuciones

Este proyecto forma parte de un sistema de detección de cáncer de piel utilizando técnicas de deep learning. El preprocesamiento y unificación de datasets es fundamental para obtener modelos robustos y precisos en el diagnóstico dermatológico asistido por IA.

---

*Fecha de última actualización:24 de Septiembre 2025*
*Versión: 1.0*
