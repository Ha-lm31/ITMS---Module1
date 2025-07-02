from ultralytics import YOLO
import cv2

# Charger le modèle
model = YOLO("yolov8s.pt")  # Tu peux changer vers yolov8s.pt, etc.

# Chemin de la vidéo d'entrée
video_path = r"C:\Users\ot\Desktop\ITMS - Module1\test4.1.mp4"

# Ouvrir la vidéo
cap = cv2.VideoCapture(video_path)

# Obtenir les dimensions de la vidéo
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Préparer l'enregistrement de la vidéo annotée
output_path = "annotated_video.mp4"
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

# Lire et traiter chaque frame
frame_count = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Appliquer YOLO sur la frame
    results = model(frame, verbose=False)

    # Annoter la frame
    annotated_frame = results[0].plot()

    # Écrire la frame annotée
    out.write(annotated_frame)

    frame_count += 1
    print(f"Frame {frame_count} traitée")

# Libérer les ressources
cap.release()
out.release()
cv2.destroyAllWindows()

print(f"Vidéo annotée enregistrée ici : {output_path}")
