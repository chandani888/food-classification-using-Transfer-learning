import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import pandas as pd
import os
import json

# ================= STEP 1: LOAD NUTRITION DATA =================
csv_file = 'indian_food_nutrition_dataset.csv'

if not os.path.exists(csv_file):
    print(f"❌ Error: {csv_file} not found!")
    exit()

# Load CSV and clean header spaces
df = pd.read_csv(csv_file)
df.columns = df.columns.str.strip()

# ================= STEP 2: LOAD CLASS NAMES =================
if not os.path.exists("classes.json"):
    print("❌ Error: classes.json not found!")
    exit()

with open("classes.json", "r") as f:
    class_names = json.load(f)

num_classes = len(class_names)

# ================= STEP 3: LOAD MODEL =================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ✅ Use MobileNetV2 with updated 'weights' parameter to remove warnings
model = models.mobilenet_v2(weights=None)

# Adjust the classifier to match your 96 classes
model.classifier[1] = nn.Sequential(
    nn.Dropout(0.3),
    nn.Linear(model.last_channel, num_classes)
)

# Load weights (Ensure food_model.pth is in the same folder)
if os.path.exists("food_model.pth"):
    model.load_state_dict(torch.load("food_model.pth", map_location=device, weights_only=True))
    model = model.to(device)
    model.eval()
    print("✅ Model loaded successfully")
else:
    print("❌ Error: food_model.pth not found!")
    exit()

# ================= STEP 4: HELPER FUNCTIONS =================

def simplify_text(text):
    """ Converts 'Dal_Makhani' or 'Dal Makhani' into 'dalmakhani' for matching. """
    return str(text).lower().replace('_', '').replace(' ', '').strip()

def analyze_food(image_path):
    # 1. Preprocess Image
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    image = Image.open(image_path).convert('RGB')
    input_tensor = transform(image).unsqueeze(0).to(device)

    # 2. Prediction
    with torch.no_grad():
        outputs = model(input_tensor)
        _, predicted_idx = torch.max(outputs, 1)
        label = class_names[predicted_idx.item()]

    # 3. Nutrition Lookup (Robust Matching)
    # Mapping for special cases where names are completely different
    mapping = {
        'chapati': 'Roti',
        'paani_puri': 'Pani Puri',
        'pakode': 'Pakora'
    }
    
    # Get the name to search for
    search_name = mapping.get(label, label)
    search_simple = simplify_text(search_name)

    # Create a simplified temporary column in the CSV to find the match
    # This matches "dal_makhani" to "Dal Makhani" perfectly
    nutrition = df[df['Food Name'].apply(simplify_text) == search_simple]

    # If no exact match, try a broader keyword search as a fallback
    if nutrition.empty:
        keyword = label.replace('_', ' ').lower()
        nutrition = df[df['Food Name'].str.lower().str.contains(keyword, na=False)]

    # ================= OUTPUT =================
    print("\n" + "=" * 40)
    print(f"🍽️  AI PREDICTION: {label.upper()}")
    print("-" * 40)

    if not nutrition.empty:
        row = nutrition.iloc[0]
        print(f"✅ CSV Match Found: {row['Food Name']}")
        print(f"🔥 Calories: {row['Calories (kcal)']} kcal")
        print(f"💪 Protein: {row['Protein (g)']} g")
        print(f"🍞 Carbs: {row['Carbohydrates (g)']} g")
        print(f"🧈 Fats: {row['Fats (g)']} g")
        print(f"🍴 Serving: {row['Serving Size']}")
    else:
        print(f"⚠️  Nutrition data for '{label}' not found in CSV.")
        print(f"💡 Suggestion: Ensure '{label.replace('_', ' ').title()}' is in your CSV.")

    print("=" * 40 + "\n")

# ================= STEP 5: RUN TEST =================
if __name__ == "__main__":
    img_path = "test3.jpg"  # Make sure this image is in your folder

    if os.path.exists(img_path):
        analyze_food(img_path)
    else:
        print(f"❌ Image not found: {img_path}")