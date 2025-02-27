import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
import cv2
from tqdm import tqdm
import logging
import matplotlib.pyplot as plt

# Logging konfigurieren
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("hand_detection.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("HandDetection")

# Erweiterte CNN-Modelldefinition
class ImprovedCNN(nn.Module):
    def __init__(self, dropout_rate=0.5):
        super(ImprovedCNN, self).__init__()
        # Mehr Filter und tieferes Netzwerk
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Berechnung der Feature-Map-Größe nach Pooling
        # Nach 3x MaxPool mit stride=2: 32x32 -> 16x16 -> 8x8 -> 4x4
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(256, 2)  # 2 Klassen: Hand und keine Hand
        )
        
        # Gewichtinitialisierung
        self._initialize_weights()
        
    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

# Erweiterte Datensatz-Klasse mit Augmentation
class HandDataset(Dataset):
    def __init__(self, hand_dir, no_hand_dir, transform=None, augment_transform=None):
        self.hand_images = [os.path.join(hand_dir, file) for file in os.listdir(hand_dir) 
                          if file.lower().endswith(('.png', '.jpg', '.jpeg'))]
        self.no_hand_images = [os.path.join(no_hand_dir, file) for file in os.listdir(no_hand_dir) 
                             if file.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        logger.info(f"Gefunden: {len(self.hand_images)} Hand-Bilder, {len(self.no_hand_images)} Nicht-Hand-Bilder")
        
        self.all_images = self.hand_images + self.no_hand_images
        self.labels = [1] * len(self.hand_images) + [0] * len(self.no_hand_images)
        self.transform = transform
        self.augment_transform = augment_transform

    def __len__(self):
        return len(self.all_images)

    def __getitem__(self, idx):
        try:
            image_path = self.all_images[idx]
            image = Image.open(image_path).convert('RGB')
            label = self.labels[idx]
            
            # Anwenden der Standard-Transformation
            if self.transform:
                image = self.transform(image)
                
            # Anwenden der Augmentation nur für Trainingsbilder und mit 50% Wahrscheinlichkeit
            if self.augment_transform and np.random.rand() > 0.5:
                image = self.augment_transform(image)
                
            return image, label
        except Exception as e:
            logger.error(f"Fehler beim Laden des Bildes {image_path}: {e}")
            # Fallback: Gib ein schwarzes Bild zurück
            if self.transform:
                dummy_image = torch.zeros(3, 32, 32)
                return dummy_image, label
            else:
                dummy_image = Image.new('RGB', (32, 32), color='black')
                return dummy_image, label

# Training-Funktionen
def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs, device, save_path):
    best_val_acc = 0.0
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]")
        for inputs, labels in progress_bar:
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # Backward + optimize
            loss.backward()
            optimizer.step()
            
            # Statistiken
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            progress_bar.set_postfix(loss=loss.item(), acc=100.*correct/total)
        
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = 100. * correct / total
        train_losses.append(epoch_loss)
        train_accuracies.append(epoch_acc)
        
        # Validation
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            progress_bar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]")
            for inputs, labels in progress_bar:
                inputs, labels = inputs.to(device), labels.to(device)
                
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                progress_bar.set_postfix(loss=loss.item(), acc=100.*correct/total)
        
        val_loss = val_loss / len(val_loader.dataset)
        val_acc = 100. * correct / total
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)
        
        # Lernrate anpassen
        scheduler.step(val_loss)
        
        logger.info(f'Epoch {epoch+1}/{num_epochs}: '
                   f'Train Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%, '
                   f'Val Loss: {val_loss:.4f}, Accuracy: {val_acc:.2f}%')
        
        # Speichere das beste Modell basierend auf Validierungsgenauigkeit
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss,
            }, save_path)
            logger.info(f'Checkpoint gespeichert: {save_path} mit Val Acc: {val_acc:.2f}%')
    
    # Plot Training-Verlauf
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Training')
    plt.plot(val_losses, label='Validation')
    plt.title('Loss über Epochen')
    plt.xlabel('Epochen')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label='Training')
    plt.plot(val_accuracies, label='Validation')
    plt.title('Genauigkeit über Epochen')
    plt.xlabel('Epochen')
    plt.ylabel('Genauigkeit (%)')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    
    return train_losses, val_losses, train_accuracies, val_accuracies

# Funktion für Echtzeit-Erkennung
def run_detection(model, transform, device, confidence_threshold=0.7):
    model.eval()
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        logger.error("Fehler beim Öffnen der Kamera")
        return
    
    logger.info("Kamera geöffnet. Drücke 'q' zum Beenden.")
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                logger.error("Fehler beim Lesen des Frames")
                break
            
            # Kopie des Frames für die Anzeige
            display_frame = frame.copy()
            
            # Bild für Modell vorbereiten
            image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            image = transform(image).unsqueeze(0).to(device)
            
            # Modellvorhersage
            with torch.no_grad():
                output = model(image)
                probabilities = torch.nn.functional.softmax(output, dim=1)
                confidence, predicted = torch.max(probabilities, 1)
                confidence = confidence.item()
                
            # Anzeigen der Ergebnisse auf dem Bild
            label = "Hand" if predicted.item() == 1 else "Keine Hand"
            color = (0, 255, 0) if predicted.item() == 1 and confidence > confidence_threshold else (0, 0, 255)
            
            # Nur als Hand markieren, wenn die Konfidenz über dem Schwellenwert liegt
            if predicted.item() == 1 and confidence > confidence_threshold:
                # Rahmen um das Bild zeichnen
                cv2.rectangle(display_frame, (0, 0), (frame.shape[1], frame.shape[0]), color, 3)
            
            # Konfidenzanzeige
            text = f"{label}: {confidence:.2f}"
            cv2.putText(display_frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            
            # Zeige das Bild an
            cv2.imshow('Hand Detection', display_frame)
            
            # Beenden mit 'q' Taste
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
    except Exception as e:
        logger.error(f"Fehler während der Erkennung: {e}")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        logger.info("Kamera geschlossen")

# Hauptfunktion
def main():
    parser = argparse.ArgumentParser(description="Hand Detection mit PyTorch")
    parser.add_argument('--data_dir', type=str, default='.', help='Verzeichnis mit den Unterordnern hand_images und no_hand_images')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch-Größe für Training')
    parser.add_argument('--epochs', type=int, default=50, help='Anzahl der Trainings-Epochen')
    parser.add_argument('--lr', type=float, default=0.001, help='Lernrate')
    parser.add_argument('--model_path', type=str, default='hand_detection_model.pth', help='Pfad zum Speichern/Laden des Modells')
    parser.add_argument('--mode', type=str, choices=['train', 'detect'], required=True, help='Modus: train oder detect')
    parser.add_argument('--val_split', type=float, default=0.2, help='Anteil der Validierungsdaten (0-1)')
    args = parser.parse_args()
    
    # CUDA verfügbar?
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info(f"Verwendetes Gerät: {device}")
    
    # Pfade
    hand_dir = os.path.join(args.data_dir, 'hand_images')
    no_hand_dir = os.path.join(args.data_dir, 'no_hand_images')
    
    # Stellen Sie sicher, dass die Verzeichnisse existieren
    if not os.path.exists(hand_dir) or not os.path.exists(no_hand_dir):
        logger.error(f"Verzeichnisse nicht gefunden: {hand_dir} oder {no_hand_dir}")
        return
    
    # Basis-Transformationen
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Augmentation-Transformationen für Training
    augment_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
    ])
    
    # Modell initialisieren
    model = ImprovedCNN().to(device)
    logger.info(f"Modell initialisiert: {model.__class__.__name__}")
    
    if args.mode == 'train':
        logger.info("Modus: Training")
        
        # Datensatz und DataLoader mit Augmentation für Training
        full_dataset = HandDataset(hand_dir, no_hand_dir, transform=transform, augment_transform=augment_transform)
        
        # Aufteilung in Trainings- und Validierungsdaten
        val_size = int(args.val_split * len(full_dataset))
        train_size = len(full_dataset) - val_size
        train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
        
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
        
        logger.info(f"Datensätze erstellt: {train_size} Training, {val_size} Validierung")
        
        # Verlustfunktion und Optimierer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
        
        # Learning Rate Scheduler
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5, verbose=True)
        
        # Training starten
        logger.info(f"Starte Training für {args.epochs} Epochen")
        train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, 
                   args.epochs, device, args.model_path)
        
        logger.info(f"Training abgeschlossen, Modell gespeichert unter {args.model_path}")
    
    elif args.mode == 'detect':
        logger.info("Modus: Erkennung")
        
        # Modell laden
        if os.path.isfile(args.model_path):
            checkpoint = torch.load(args.model_path, map_location=device)
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
                logger.info(f"Modell geladen von {args.model_path}")
                if 'val_acc' in checkpoint:
                    logger.info(f"Validierungsgenauigkeit: {checkpoint['val_acc']:.2f}%")
            else:
                model.load_state_dict(checkpoint)
                logger.info(f"Modell geladen von {args.model_path} (älteres Format)")
        else:
            logger.error(f"Kein gespeichertes Modell gefunden unter {args.model_path}")
            return
        
        # Starte Echtzeit-Erkennung
        run_detection(model, transform, device)

if __name__ == "__main__":
    main()
