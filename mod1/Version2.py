#pas utile : à modifie
#json!!

from ultralytics import YOLO
import os
from collections import defaultdict
import time

t1 = time.time()

# Charger le modèle YOLO
model = YOLO("yolov8n.pt")

# Dossier des images
image_folder = "test_images"

# Dictionnaire des classes véhicules selon COCO
vehicle_classes = {
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck"
}

# Comptage global
total_counts = defaultdict(int)

# Parcours des images
for image_file in os.listdir(image_folder):
    if image_file.lower().endswith((".jpg", ".jpeg", ".png")):
        image_path = os.path.join(image_folder, image_file)
        results = model(image_path)

        for result in results:
            # Compter les véhicules par type
            for cls_id in result.boxes.cls.tolist():
                cls_id = int(cls_id)
                if cls_id in vehicle_classes:
                    total_counts[vehicle_classes[cls_id]] += 1

            # Sauvegarder l’image avec détection
            result.save()

# Affichage des totaux
print("Nombre total de véhicules détectés :")
for v_type, count in total_counts.items():
    print(f" - {v_type}: {count}")

t2 = time.time()
print(f'Temps dèxécution {t2 - t1} unités')