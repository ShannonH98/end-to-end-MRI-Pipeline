import os
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
from preprocessing.label_map import load_labels

class BrainSliceDataset(Dataset):
    def __init__(self, slices_folder, tsv_path):
        self.samples = []
        self.transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
        ])

        labels = load_labels(tsv_path)

        for subject in os.listdir(slices_folder):
            subject_path = os.path.join(slices_folder, subject)
            if not os.path.isdir(subject_path):
                continue

            # Extract participant ID from folder name e.g. sub-01
            pid = "sub-" + subject.split("-")[1].split("_")[0]

            if pid not in labels:
                print(f"Skipping {subject} — not in participants.tsv")
                continue

            label = labels[pid]

            for slice_file in sorted(os.listdir(subject_path)):
                if slice_file.endswith(".png"):
                    self.samples.append((
                        os.path.join(subject_path, slice_file),
                        label
                    ))

        print(f"Dataset ready: {len(self.samples)} slices")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("L")
        img = self.transform(img)
        return img, label