# import os
# import torchaudio

# def convert_to_16kHz(input_folder, output_folder):
#     os.makedirs(output_folder, exist_ok=True)

#     for filename in os.listdir(input_folder):
#         if filename.endswith((".wav", ".mp3")):
#             filepath = os.path.join(input_folder, filename)
#             output_path = os.path.join(output_folder, filename.replace(".mp3", ".wav"))

#             # Load audio
#             speech_array, sample_rate = torchaudio.load(filepath)
            
#             # Convert if necessary
#             if sample_rate != 16000:
#                 resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
#                 speech_array = resampler(speech_array)

#             # Save standardized audio
#             torchaudio.save(output_path, speech_array, 16000)
#             print(f"✅ Converted {filename} to 16kHz")

# # Apply conversion for both REAL & FAKE folders
# convert_to_16kHz("dataset_audio/REAL", "dataset_audio/REAL_16kHz")
# convert_to_16kHz("dataset_audio/FAKE", "dataset_audio/FAKE_16kHz")

# print("🎯 All files are now in 16kHz format!")






# import torch
# import torch.nn as nn
# import torch.optim as optim
# from transformers import Wav2Vec2Processor, Wav2Vec2Model

# # Set Device
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(f"Using device: {device}")

# # Load Pretrained Wav2Vec2 Model & Processor
# processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
# base_model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h").to(device)

# # Define Custom Classifier
# class Wav2Vec2Classifier(nn.Module):
#     def __init__(self):
#         super(Wav2Vec2Classifier, self).__init__()
#         self.wav2vec2 = base_model  # Keep base model
#         self.fc = nn.Linear(self.wav2vec2.config.hidden_size, 2)  # Binary classification (Real vs Fake)

#     def forward(self, input_values):
#         with torch.no_grad():
#             outputs = self.wav2vec2(input_values).last_hidden_state.mean(dim=1)  # Feature extraction

#         return self.fc(outputs)

# # Initialize Classifier Model
# classifier_model = Wav2Vec2Classifier().to(device)

# # Training Setup
# criterion = nn.CrossEntropyLoss()
# optimizer = optim.Adam(classifier_model.parameters(), lr=1e-5)

# # Example Training Loop (Modify for actual dataset)
# def train_classifier(train_loader):
#     classifier_model.train()
    
#     for batch in train_loader:
#         inputs, labels = batch
#         inputs, labels = inputs.to(device), labels.to(device)

#         optimizer.zero_grad()
#         outputs = classifier_model(inputs)
#         loss = criterion(outputs, labels)
#         loss.backward()
#         optimizer.step()

#         print(f"Loss: {loss.item()}")

# # Save Model After Training
# torch.save(classifier_model.state_dict(), "wav2vec2_deepfake_classifier.pth")
# print("✅ Model trained and saved!")




# import torchaudio
# from torch.utils.data import Dataset, DataLoader
# import os
# from transformers import Wav2Vec2Processor
# import torch

# processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")

# class DeepfakeAudioDataset(Dataset):
#     def __init__(self, real_folder, fake_folder, processor):
#         self.filepaths = [(os.path.join(real_folder, f), 1) for f in os.listdir(real_folder)] + \
#                          [(os.path.join(fake_folder, f), 0) for f in os.listdir(fake_folder)]
#         self.processor = processor

#     def __len__(self):
#         return len(self.filepaths)

#     def __getitem__(self, idx):
#         filepath, label = self.filepaths[idx]
#         speech_array, sample_rate = torchaudio.load(filepath)
#         speech_array = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)(speech_array).squeeze()

#         inputs = self.processor(speech_array, sampling_rate=16000, return_tensors="pt", padding=True)
#         return inputs.input_values.squeeze(0), torch.tensor(label)

# # Create DataLoader
# train_dataset = DeepfakeAudioDataset("dataset_audio/REAL_16kHz", "dataset_audio/FAKE_16kHz", processor)
# train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)

# print(f"✅ DataLoader prepared with {len(train_dataset)} samples")




# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import DataLoader, Dataset
# from transformers import Wav2Vec2Processor, Wav2Vec2Model
# import torchaudio
# import os

# # Set device
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(f"Using device: {device}")

# # Load Wav2Vec2 processor & base model
# processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
# base_model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h").to(device)

# # Define Custom Classifier
# class Wav2Vec2Classifier(nn.Module):
#     def __init__(self):
#         super(Wav2Vec2Classifier, self).__init__()
#         self.wav2vec2 = base_model
#         self.fc = nn.Linear(self.wav2vec2.config.hidden_size, 2)  # Binary classification (Real vs Fake)

#     def forward(self, input_values):
#         with torch.no_grad():
#             outputs = self.wav2vec2(input_values).last_hidden_state  # Extract features
#         return self.fc(outputs.mean(dim=1))  # Average across time dimension

# # Initialize classifier model
# classifier_model = Wav2Vec2Classifier().to(device)

# # Define optimizer & loss function
# optimizer = optim.Adam(classifier_model.parameters(), lr=1e-5)
# criterion = nn.CrossEntropyLoss()

# # Define dataset structure
# class DeepfakeAudioDataset(Dataset):
#     def __init__(self, real_dir, fake_dir, processor):
#         self.filepaths = [(os.path.join(real_dir, f), 1) for f in os.listdir(real_dir)] + \
#                          [(os.path.join(fake_dir, f), 0) for f in os.listdir(fake_dir)]
#         self.processor = processor
    
#     def __len__(self):
#         return len(self.filepaths)

#     def __getitem__(self, idx):
#         filepath, label = self.filepaths[idx]
#         speech_array, sample_rate = torchaudio.load(filepath)

#         # Ensure consistent 16kHz resampling
#         resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
#         speech_array = resampler(speech_array)

#         # Force conversion to **MONO** by averaging stereo channels
#         if speech_array.shape[0] == 2:  # If stereo, convert to mono
#             speech_array = speech_array.mean(dim=0)

#         speech_array = speech_array.squeeze()  # Remove extra batch dimension if present

#         # Process audio using Wav2Vec2
#         inputs = self.processor(speech_array, sampling_rate=16000, return_tensors="pt", padding=True)
#         input_values = inputs.input_values.squeeze(0)  # Fix tensor shape

#         return input_values, torch.tensor(label, dtype=torch.long)

# # Function to correctly batch inputs & labels
# def collate_fn(batch):
#     # Ensure all tensors are 1D before padding
#     inputs = [b[0].squeeze() if b[0].dim() > 1 else b[0] for b in batch]
    
#     # Find longest sample and pad others to match
#     max_length = max(inp.shape[-1] for inp in inputs)
#     inputs = [torch.nn.functional.pad(inp, (0, max_length - inp.shape[-1])) for inp in inputs]
#     inputs = torch.stack(inputs)  # Stack into batch format
    
#     labels = torch.tensor([b[1] for b in batch])  # Convert labels to tensor
#     return inputs, labels



# # Initialize dataset and DataLoader
# train_dataset = DeepfakeAudioDataset("dataset_audio/REAL_16kHz", "dataset_audio/FAKE_16kHz", processor)
# train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, collate_fn=collate_fn)

# # Check dataset integrity
# if len(train_dataset) == 0:
#     raise ValueError("❌ Dataset is empty! Check if audio files exist in REAL_16kHz & FAKE_16kHz.")

# # Training loop
# for epoch in range(5):  # Adjust epochs based on performance
#     classifier_model.train()
#     total_loss = 0

#     for batch in train_loader:
#         inputs, labels = batch
#         inputs, labels = inputs.to(device), labels.to(device)

#         optimizer.zero_grad()
#         outputs = classifier_model(inputs)
#         loss = criterion(outputs, labels)
#         loss.backward()
#         optimizer.step()
        
#         total_loss += loss.item()

#     print(f"Epoch {epoch+1} Loss: {total_loss / len(train_loader)}")

# # Save trained model
# torch.save(classifier_model.state_dict(), "wav2vec2_deepfake_classifier.pth")
# print("✅ Model training complete and saved!")



import torch
import torchaudio
from transformers import Wav2Vec2Processor, Wav2Vec2Model
import torch.nn as nn

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load processor & base Wav2Vec2 model
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
base_model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h").to(device)

# ✅ Define classifier before loading its weights
class Wav2Vec2Classifier(nn.Module):
    def __init__(self):
        super(Wav2Vec2Classifier, self).__init__()
        self.wav2vec2 = base_model
        self.fc = nn.Linear(self.wav2vec2.config.hidden_size, 2)  # Binary classification (Real vs Fake)

    def forward(self, input_values):
        with torch.no_grad():
            outputs = self.wav2vec2(input_values).last_hidden_state  # Extract features
        return self.fc(outputs.mean(dim=1))  # Average across time dimension

# ✅ Initialize classifier model BEFORE loading weights
classifier_model = Wav2Vec2Classifier().to(device)
classifier_model.load_state_dict(torch.load("wav2vec2_deepfake_classifier.pth"))
classifier_model.eval()  # Set model to evaluation mode

# Function to classify deepfake audio from direct file input
def detect_audio_deepfake(file_path):
    speech_array, sample_rate = torchaudio.load(file_path)

    # Ensure 16kHz resampling
    resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
    speech_array = resampler(speech_array).mean(dim=0).squeeze()  # Convert stereo to mono

    # Ensure correct input shape: [batch_size, sequence_length]
    inputs = processor(speech_array, sampling_rate=16000, return_tensors="pt", padding=True)
    input_values = inputs.input_values.to(device)

    # **Fix dimension mismatch**
    input_values = input_values.squeeze().unsqueeze(0)  # Ensure single batch format

    # Run inference
    with torch.no_grad():
        outputs = classifier_model(input_values)
        confidence = torch.nn.functional.softmax(outputs, dim=1)[0][1].item()  # Probability of being real

    is_deepfake = confidence < 0.5
    return {
        'file': file_path,
        'is_deepfake': is_deepfake,
        'confidence': round((1 - confidence if is_deepfake else confidence) * 100, 2),
        'verdict': 'Deepfake' if is_deepfake else 'Genuine'
    }


# ✅ Example usage: Direct file path input
file_path = r"C:\Users\bhoom\Downloads\file10 (1).wav"  # Replace with actual audio file path
result = detect_audio_deepfake(file_path)
print(result)
