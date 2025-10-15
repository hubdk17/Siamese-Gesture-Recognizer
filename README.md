import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import mediapipe as mp
import os

# ------------------------------
# 1. Siamese Encoder Definition
# ------------------------------
class SiameseEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(128, 64)
        )

    def forward_once(self, x):
        x = self.cnn(x)
        return F.normalize(x, p=2, dim=1)

# ------------------------------
# 2. Hand Extraction via Mediapipe
# ------------------------------
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

def extract_hand(frame):
    """Returns cropped grayscale hand region or None."""
    with mp_hands.Hands(static_image_mode=True,
                        max_num_hands=1,
                        min_detection_confidence=0.5) as hands:
        results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if results.multi_hand_landmarks:
            h, w, _ = frame.shape
            coords = [(int(lm.x * w), int(lm.y * h)) for lm in results.multi_hand_landmarks[0].landmark]
            x_min, y_min = np.min(coords, axis=0)
            x_max, y_max = np.max(coords, axis=0)
            pad = 20
            x_min, y_min = max(0, x_min - pad), max(0, y_min - pad)
            x_max, y_max = min(w, x_max + pad), min(h, y_max + pad)
            hand_img = frame[y_min:y_max, x_min:x_max]
            hand_gray = cv2.cvtColor(hand_img, cv2.COLOR_BGR2GRAY)
            hand_resized = cv2.resize(hand_gray, (100, 100))
            return hand_resized
        return None

# ------------------------------
# 3. Initialize Model & Reference Embeddings
# ------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
model = SiameseEncoder().to(device)
model.eval()

ref_dir = "reference_gestures"
os.makedirs(ref_dir, exist_ok=True)

# ------------------------------
# 4. Helper: Compute Embedding
# ------------------------------
def get_embedding(img_gray):
    img_tensor = torch.tensor(img_gray / 255.0, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
    with torch.no_grad():
        emb = model.forward_once(img_tensor)
    return emb.cpu().numpy().flatten()

# ------------------------------
# 5. Record Reference Gestures
# ------------------------------
def record_references():
    cap = cv2.VideoCapture(0)
    gesture_name = input("Enter gesture name to record (e.g., 'fist'): ").strip()
    save_path = os.path.join(ref_dir, f"{gesture_name}.npy")

    print("[Press SPACE to capture 5 reference images, 'q' or '1' to quit]")
    refs = []
    count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        hand = extract_hand(frame)
        if hand is not None:
            cv2.imshow("Hand", hand)
        cv2.imshow("Webcam", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == 32 and hand is not None:  # SPACE to capture
            emb = get_embedding(hand)
            refs.append(emb)
            count += 1
            print(f"Captured {count}/5")

        elif key in [ord('q'), ord('1'), 27]:  # q, 1, or ESC to stop
            print("Stopping capture...")
            break

        if count >= 5:
            print("Captured all 5 images.")
            break

    cap.release()
    cv2.destroyAllWindows()

    if refs:
        np.save(save_path, np.array(refs))
        print(f"Saved reference embeddings to {save_path}")

# ------------------------------
# 6. Live Recognition
# ------------------------------
def live_recognition(threshold=0.7):
    # Load all reference gestures
    refs = {}
    for file in os.listdir(ref_dir):
        if file.endswith(".npy"):
            name = file[:-4]
            refs[name] = np.load(os.path.join(ref_dir, file))

    if not refs:
        print("No reference gestures found. Run record_references() first.")
        return

    cap = cv2.VideoCapture(0)
    print("Press 'q', '1', or ESC to exit.")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)

        # Extract hand
        hand = extract_hand(frame)
        gesture = "No Hand"

        if hand is not None:
            cv2.imshow("Cropped Hand", hand)
            emb = get_embedding(hand)
            best_name, best_dist = None, float("inf")

            # Compare with reference gestures
            for name, samples in refs.items():
                dists = np.linalg.norm(samples - emb, axis=1)
                mean_dist = np.mean(dists)
                print(f"Distance to {name}: {mean_dist:.4f}")  # DEBUG
                if mean_dist < best_dist:
                    best_dist, best_name = mean_dist, name

            # Decide detected gesture
            if best_dist < threshold:
                gesture = best_name
            else:
                gesture = "Unknown"

        # Display gesture on full webcam feed
        cv2.putText(frame, f"Gesture: {gesture}", (30, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Gesture Recognition", frame)

        # Exit keys
        key = cv2.waitKey(1) & 0xFF
        if key in [ord('q'), ord('1'), 27]:  # q, 1, ESC
            print("Exiting live recognition...")
            break

    cap.release()
    cv2.destroyAllWindows()





# ------------------------------
# 7. Run
# ------------------------------
if __name__ == "__main__":
    print("1. Record reference gestures")
    print("2. Start live recognition")
    choice = input("Choose (1/2): ")
    if choice == "1":
        record_references()
    else:
        live_recognition()


