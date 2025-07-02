import time
t1 = time.time()

from ultralytics import YOLO
import cv2

# Charger le modèle YOLOv8
model = YOLO("yolov8n.pt")

# Charger l'image
img_path = r"C:\Users\ot\Desktop\ITMS - Module1\E-S model\img-31.jpg"
image = cv2.imread(img_path)
height, width, _ = image.shape

# Faire l'inférence
results = model(image)[0]

# Initialiser le compteur
vehicle_count_left = 0
vehicle_count_right = 0

# Classes des véhicules dans COCO (voiture, camion, bus, moto)
vehicle_classes = [2, 3, 5, 7]

# Traiter chaque boîte détectée
for box in results.boxes:
    cls = int(box.cls.item())
    if cls in vehicle_classes:
        # Coordonnées de la boîte
        x1, y1, x2, y2 = box.xyxy[0]
        x_center = (x1 + x2) / 2

        # Déterminer le sens selon la position du centre
        if x_center < width / 2:
            vehicle_count_left += 1  # Sens gauche (peut représenter direction vers la droite)
        else:
            vehicle_count_right += 1  # Sens droit (direction vers la gauche)

# Affichage
print(f"Véhicules dans le sens gauche ➝ droite : {vehicle_count_left}")
print(f"Véhicules dans le sens droite ➝ gauche : {vehicle_count_right}")

# Afficher l'image annotée
annotated_img = results.plot()
cv2.imshow("Détection des véhicules", annotated_img)
cv2.waitKey(0)
cv2.destroyAllWindows()


t2 = time.time()
print(t2 - t1)