from ultralytics import YOLO
import os


# Charger le modèle (choisis : yolov8n.pt, yolov8s.pt, yolov8m.pt, yolov8l.pt, yolov8x.pt)
# modèle nano pour commencer
model = YOLO("yolov8n.pt")

# Dossier des images
image_folder = "test_images"

# Parcourir et détecter chaque image
for image_file in os.listdir(image_folder):
    if image_file.endswith((".jpg", ".png", ".jpeg")):
        image_path = os.path.join(image_folder, image_file)
        # 'save=True' pour enregistrer les résultats avec annotations
        results = model(image_path, save = True)
        print(f"Résultats enregistrés pour {image_file}")

'''
type-vehicule(class) : nombre
'''