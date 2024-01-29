import cv2
import os

def get_img_count(folder):
    """Zählt die Anzahl der Bilder in einem Ordner."""
    if not os.path.exists(folder):
        os.makedirs(folder)  # Erstelle den Ordner, falls er nicht existiert
        return 0  # Keine Bilder, da der Ordner neu erstellt wurde
    return len([name for name in os.listdir(folder) if os.path.isfile(os.path.join(folder, name))])

# Pfade für die Bildspeicherung
hand_dir = '/home/emilio/Dokumente/ai/frnn/hand_images'
no_hand_dir = '/home/emilio/Dokumente/ai/frnn/no_hand_images'

# Zähle vorhandene Bilder, um den img_counter zu initialisieren
img_counter_hand = get_img_count(hand_dir)
img_counter_no_hand = get_img_count(no_hand_dir)

# Kamera initialisieren
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Kann die Kamera nicht öffnen")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Fehler beim Erfassen des Frames. Schließen.")
        break

    cv2.imshow('Frame', frame)

    k = cv2.waitKey(1)

    if k % 256 == 27:  # Esc-Taste für Beenden
        print("Escape gedrückt, schließe...")
        break
    elif k % 256 == ord('d'):  # D-Taste für Hand
        img_name = os.path.join(hand_dir, f"hand_{img_counter_hand}.jpg")
        cv2.imwrite(img_name, frame)
        print(f"{img_name} gespeichert!")
        img_counter_hand += 1
    elif k % 256 == ord('f'):  # F-Taste für keine Hand
        img_name = os.path.join(no_hand_dir, f"no_hand_{img_counter_no_hand}.jpg")
        cv2.imwrite(img_name, frame)
        print(f"{img_name} gespeichert!")
        img_counter_no_hand += 1

cap.release()
cv2.destroyAllWindows()
