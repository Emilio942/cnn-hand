import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import cv2




# CNN-Modelldefinition
class SimpleCNN(nn.Module):
    # ... (Modelldefinition bleibt gleich) ...
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(64 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, 2)  # 2 Klassen: Hand und keine Hand

    def forward(self, x):
        x = nn.functional.relu(self.conv1(x))
        x = nn.functional.max_pool2d(x, kernel_size=2, stride=2)
        x = nn.functional.relu(self.conv2(x))
        x = nn.functional.max_pool2d(x, kernel_size=2, stride=2)
        x = x.view(-1, 64 * 8 * 8)
        x = nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        return x
# Datensatz-Klasse

class HandDataset(Dataset):
    # ... (Datensatzdefinition bleibt gleich) ...
    def __init__(self, hand_dir, no_hand_dir, transform=None):
        self.hand_images = [os.path.join(hand_dir, file) for file in os.listdir(hand_dir)]
        self.no_hand_images = [os.path.join(no_hand_dir, file) for file in os.listdir(no_hand_dir)]
        self.all_images = self.hand_images + self.no_hand_images
        self.labels = [1] * len(self.hand_images) + [0] * len(self.no_hand_images)
        self.transform = transform

    def __len__(self):
        return len(self.all_images)

    def __getitem__(self, idx):
        image_path = self.all_images[idx]
        image = Image.open(image_path).convert('RGB')
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label
# Pfad zum gespeicherten Modell

# Transformation für die Trainingsdaten
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
])

# Erstelle den Datensatz und DataLoader
hand_dir = '/home/emilio/Dokumente/ai/frnn/to/hand_images'
no_hand_dir = '/home/emilio/Dokumente/ai/frnn/to/no_hand_images'
dataset = HandDataset(hand_dir, no_hand_dir, transform=transform)
trainloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Modellinstanz, Verlustfunktion und Optimierer
model = SimpleCNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
model_save_path = '/home/emilio/Dokumente/ai/frnn/model.pth'

# Überprüfen, ob das gespeicherte Modell existiert
if os.path.isfile(model_save_path):
    # Lade das trainierte Modell, wenn es existiert
    model = SimpleCNN()
    model.load_state_dict(torch.load(model_save_path))
    model.eval()
    print(f"Modell geladen von {model_save_path}")
else:
    print(f"Kein gespeichertes Modell gefunden unter {model_save_path}")
    # Hier kannst du entscheiden, ob du das Modell neu trainieren möchtest
# Training des Modells
epochs = 1000
for epoch in range(epochs):
    running_loss = 0.0
    for images, labels in trainloader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f'Epoch {epoch + 1}, Loss: {running_loss / len(trainloader)}')

print('Training abgeschlossen')

# Speichere das trainierte Modell
model_save_path = '/home/emilio/Dokumente/ai/frnn/model.pth'
torch.save(model.state_dict(), model_save_path)
print(f"Modell gespeichert unter {model_save_path}")

# Kamerastream für Echtzeit-Erkennung
model.eval()  # Setze das Modell in den Evaluierungsmodus
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    
    # Bild für Modell vorbereiten
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    image = transform(image).unsqueeze(0)
    
    # Modellvorhersage
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output.data, 1)
        # Verarbeite die Ausgabe hier
        
    # Prüfe die Vorhersage und markiere das Bild
    if predicted.item() == 1:  # Klasse 'Hand'
        # Rahmen um das Bild grün zeichnen
        cv2.rectangle(frame, (0, 0), (frame.shape[1], frame.shape[0]), (0, 255, 0), 5)

    # Zeige das Bild an
    cv2.imshow('Frame', frame)
    
    # Beenden mit 'q' Taste
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
