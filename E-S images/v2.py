import time
from ultralytics import YOLO
import cv2

# Mesure de temps
t1 = time.time()

# Charger le modèle YOLOv8
model = YOLO("yolov8n.pt")

# Charger l'image
img_path = r"C:\Users\ot\Desktop\ITMS - Module1\E-S model\img-31.jpg"
image = cv2.imread(img_path)
height, width, _ = image.shape

# Faire la détection
results = model(image)[0]

# Liste des classes COCO pour les véhicules
class_names = model.names  # dict index → nom
vehicle_classes = [2, 3, 5, 7]  # voiture, moto, bus, camion

# Compteurs par type de véhicule (arrivants uniquement)
vehicle_counts = {class_names[i]: 0 for i in vehicle_classes}

# Traitement des détections
for box in results.boxes:
    cls = int(box.cls.item())
    if cls in vehicle_classes:
        x1, y1, x2, y2 = box.xyxy[0]
        x_center = (x1 + x2) / 2

        # Garder uniquement les véhicules dans la moitié gauche (qui arrivent)
        if x_center < width / 2:
            vehicle_counts[class_names[cls]] += 1

# Affichage des résultats
print("Véhicules ARRIVANTS par type :")
for veh_type, count in vehicle_counts.items():
    print(f" - {veh_type} : {count}")

# Annoter et dessiner la ligne centrale
annotated_img = results.plot()
cv2.line(annotated_img, (width // 2, 0), (width // 2, height), (0, 255, 0), 2)

# Affichage de l'image annotée
cv2.imshow("Détection - Véhicules ARRIVANTS", annotated_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Temps total
t2 = time.time()
print(f"Temps d'exécution : {round(t2 - t1, 2)} secondes")
