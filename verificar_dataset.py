"""
Script de verificación rápida del procesamiento de imágenes
"""

from pathlib import Path

# Verificar estructura
BASE_DIR = Path("Datos/Dataset_Evolutivo_Simulado")
INPUT_DIR = BASE_DIR / "train_B"
OUTPUT_DIR = BASE_DIR / "train_B_1"

print("="*60)
print("🔍 VERIFICACIÓN DEL DATASET")
print("="*60)

# Verificar que existan las carpetas
print(f"\n📁 Carpetas:")
print(f"   - INPUT (train_B): {'✅ Existe' if INPUT_DIR.exists() else '❌ No existe'}")
print(f"   - OUTPUT (train_B_1): {'✅ Existe' if OUTPUT_DIR.exists() else '❌ No existe'}")

# Contar imágenes
if INPUT_DIR.exists():
    input_images = list(INPUT_DIR.glob("*.jpg")) + list(INPUT_DIR.glob("*.png"))
    print(f"\n📊 Imágenes en train_B: {len(input_images)}")
else:
    input_images = []
    print(f"\n⚠️ No se puede contar imágenes en train_B (carpeta no existe)")

if OUTPUT_DIR.exists():
    output_images = list(OUTPUT_DIR.glob("*.jpg")) + list(OUTPUT_DIR.glob("*.png"))
    print(f"📊 Imágenes en train_B_1: {len(output_images)}")

    if len(output_images) == 0:
        print("\n⚠️ La carpeta train_B_1 está vacía!")
        print("   → Ejecuta las celdas #7 y #8 del notebook para procesar las imágenes")
    elif len(input_images) > 0 and len(output_images) < len(input_images):
        print(f"\n⚠️ Faltan {len(input_images) - len(output_images)} imágenes por procesar")
        print("   → Ejecuta nuevamente la celda #8 del notebook")
    elif len(input_images) > 0 and len(output_images) == len(input_images):
        print("\n✅ ¡Perfecto! Todas las imágenes han sido procesadas correctamente")
else:
    print(f"\n⚠️ La carpeta train_B_1 no existe aún")
    print("   → Ejecuta la celda #2 del notebook para crearla")

print("\n" + "="*60)
print("📝 INSTRUCCIONES:")
print("="*60)
print("""
Para procesar todas las imágenes, ejecuta en orden:

1. Celda #1: Importar librerías
2. Celda #2: Configurar rutas (crea train_B_1)
3. Celda #3-5: Definir funciones
6. Celda #6: Visualizar ejemplos (opcional)
7. Celda #7: Configurar procesamiento
8. Celda #8: ¡PROCESAR TODAS LAS IMÁGENES! ← IMPORTANTE
9. Celda #9: Verificar resultados
10. Celda #10: Ver mosaico final
11. Celda #11: Resumen

Después podrás subir a Google Drive:
- train_B/ (originales limpias - tiempo t=0)
- train_B_1/ (con manchas simuladas - tiempo t=6 meses)
""")
print("="*60)

