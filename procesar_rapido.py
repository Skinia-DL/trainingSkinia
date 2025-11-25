"""
Script RÁPIDO para procesar imágenes por lotes
Ejecuta esto si el notebook tarda mucho
"""

import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import random
from scipy.ndimage import gaussian_filter

# Configuración
BASE_DIR = Path("Datos/Dataset_Evolutivo_Simulado")
INPUT_DIR = BASE_DIR / "train_B"
OUTPUT_DIR = BASE_DIR / "train_B_1"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

EVOLUTION_MONTHS = 6
BATCH_SIZE = 50  # Procesa de 50 en 50

# Funciones optimizadas (versión rápida)
def generate_irregular_mask_fast(img_shape, num_spots=5):
    """Versión rápida de máscaras irregulares"""
    h, w = img_shape[:2]
    mask = np.zeros((h, w), dtype=np.float32)

    for _ in range(num_spots):
        cx = random.randint(60, w - 60)
        cy = random.randint(60, h - 60)
        radius = random.randint(30, 80)

        y, x = np.ogrid[:h, :w]
        circle_mask = ((x - cx)**2 + (y - cy)**2 <= radius**2).astype(np.float32)
        mask = np.maximum(mask, circle_mask)

    mask = gaussian_filter(mask, sigma=5)
    return mask

def apply_fast_simulation(image):
    """Simulación rápida de manchas"""
    img = image.copy().astype(np.float32) / 255.0

    # Máscara simple
    mask = generate_irregular_mask_fast(img.shape, num_spots=random.randint(3, 7))

    # Niebla rápida
    h, w = img.shape[:2]
    fog = np.random.randn(h, w) * 0.15
    fog = gaussian_filter(fog, sigma=10)

    mask_3d = np.stack([mask] * 3, axis=-1)
    fog_3d = np.stack([fog] * 3, axis=-1)

    # Aplicar efectos
    result = img + fog_3d * mask_3d * 0.3

    # Oscurecer
    darkening = 1.0 - (mask_3d * 0.15)
    result = result * darkening

    result = np.clip(result, 0, 1)
    return (result * 255).astype(np.uint8)

# Obtener imágenes
image_files = list(INPUT_DIR.glob("*.jpg")) + list(INPUT_DIR.glob("*.png"))
print(f"📊 Total de imágenes: {len(image_files)}")

# Verificar cuántas ya están procesadas
existing = list(OUTPUT_DIR.glob("*.jpg")) + list(OUTPUT_DIR.glob("*.png"))
existing_names = {f.name for f in existing}
pending = [f for f in image_files if f.name not in existing_names]

print(f" Ya procesadas: {len(existing)}")
print(f" Pendientes: {len(pending)}")

if len(pending) == 0:
    print("\n¡Todas las imágenes ya están procesadas!")
else:
    # Procesar por lotes
    print(f"\n Procesando en lotes de {BATCH_SIZE}...")
    print(f" Tiempo estimado: ~{len(pending) * 0.15 / 60:.1f} minutos\n")

    random.seed(42)
    np.random.seed(42)

    success = 0
    errors = 0

    for idx, img_path in enumerate(tqdm(pending, desc="Procesando")):
        try:
            # Leer y procesar
            image = cv2.imread(str(img_path))
            if image is None:
                errors += 1
                continue

            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            simulated = apply_fast_simulation(image)
            simulated = cv2.cvtColor(simulated, cv2.COLOR_RGB2BGR)

            # Guardar
            output_path = OUTPUT_DIR / img_path.name
            cv2.imwrite(str(output_path), simulated, [cv2.IMWRITE_JPEG_QUALITY, 95])
            success += 1

            # Checkpoint
            if (idx + 1) % BATCH_SIZE == 0:
                print(f"\n✓ Procesadas: {success + len(existing)}/{len(image_files)}")

        except Exception as e:
            errors += 1
            print(f"\n Error en {img_path.name}: {e}")

    print(f"\n{'='*60}")
    print(f" Procesamiento completado!")
    print(f"   - Total procesadas: {success + len(existing)}/{len(image_files)}")
    print(f"   - Errores: {errors}")
    print(f"{'='*60}")

# Verificación final
final_count = len(list(OUTPUT_DIR.glob("*.jpg")) + list(OUTPUT_DIR.glob("*.png")))
print(f"\nImágenes finales en train_B_1: {final_count}")

if final_count == len(image_files):
    print("¡PERFECTO! Todas las imágenes procesadas correctamente")
    print("\nSiguiente paso: Subir a Google Drive")
    print(f"   - train_B/ (originales)")
    print(f"   - train_B_1/ (con manchas)")
else:
    print(f"Faltan {len(image_files) - final_count} imágenes")
    print("   → Ejecuta el script nuevamente para continuar")

