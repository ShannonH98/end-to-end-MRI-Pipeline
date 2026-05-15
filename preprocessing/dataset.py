import os
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
from preprocessing.label_map import load_labels

class BrainSliceDataset(Dataset):
    def __init__(self, slices_folder, tsv_path, subject_ids=None, include_augmented=True):
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

            pid = "sub-" + subject.split("-")[1].split("_")[0]

            if pid not in labels:
                print(f"Skipping {subject} — not in participants.tsv")
                continue

            if subject_ids is not None and pid not in subject_ids:
                continue

            label = labels[pid]

            orig_slices = sorted([f for f in os.listdir(subject_path) if f.endswith(".png") and "_aug" not in f])
            n = len(orig_slices)
            start, end = int(0.25 * n), int(0.75 * n)
            selected = set(orig_slices[start:end])

            for slice_file in sorted(os.listdir(subject_path)):
                if not slice_file.endswith(".png"):
                    continue
                if not include_augmented and "_aug" in slice_file:
                    continue
                base = slice_file.split("_aug")[0] + ".png" if "_aug" in slice_file else slice_file
                if base not in selected:
                    continue
                self.samples.append((
                    os.path.join(subject_path, slice_file),
                    label,
                    pid
                ))

        print(f"Dataset ready: {len(self.samples)} slices")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label, subject_id = self.samples[idx]
        img = Image.open(path).convert("L")
        img = self.transform(img)
        return img, label, subject_id
