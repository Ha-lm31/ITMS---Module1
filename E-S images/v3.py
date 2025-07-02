import time
from ultralytics import YOLO
import cv2

# Début du chrono
t1 = time.time()

# Charger le modèle YOLOv8
model = YOLO("yolov8m.pt")

# Chemin vers la vidéo d'entrée
video_path = r"C:\Users\ot\Desktop\ITMS - Module1\test4.1.mp4"
cap = cv2.VideoCapture(video_path)

# Vérifier si la vidéo est chargée
if not cap.isOpened():
    print("Erreur : Impossible d'ouvrir la vidéo.")
    exit()

# Récupérer les dimensions et le FPS
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# Configuration de la vidéo de sortie
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter("video_sortie_comptage.mp4", fourcc, fps, (width, height))

# Classes de véhicules à détecter
class_names = model.names
vehicle_classes = [2, 3, 5, 7]  # voiture, moto, bus, camion

# Boucle sur chaque frame
while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)[0]

    # Initialiser les compteurs
    vehicle_counts = {class_names[i]: 0 for i in vehicle_classes}

    # Boucle sur chaque détection
    for box in results.boxes:
        cls = int(box.cls.item())
        if cls in vehicle_classes:
            x1, y1, x2, y2 = box.xyxy[0]
            x_center = (x1 + x2) / 2

            # Compter seulement les véhicules sur la moitié gauche
            if x_center < width / 2:
                vehicle_counts[class_names[cls]] += 1

    # Annoter l'image
    annotated_frame = results.plot()

    # Ligne verte au centre
    cv2.line(annotated_frame, (width // 2, 0), (width // 2, height), (0, 255, 0), 2)

    # Affichage des compteurs sur l’image
    y_offset = 30
    for veh_type, count in vehicle_counts.items():
        text = f"{veh_type}: {count}"
        cv2.putText(annotated_frame, text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        y_offset += 30

    # Sauvegarde de la frame annotée
    out.write(annotated_frame)

    # Affichage en temps réel (optionnel)
    cv2.imshow("Détection véhicules", annotated_frame)
    if cv2.waitKey(1) & 0xFF == 27:  # Appuie sur 'Échap' pour arrêter
        break

# Nettoyage
cap.release()
out.release()
cv2.destroyAllWindows()

# Temps d'exécution
t2 = time.time()
print(f"Vidéo traitée. Temps total : {round(t2 - t1, 2)} secondes")