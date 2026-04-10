import csv

def load_labels(tsv_path):
    labels = {}
    label_map = {"HC": 0, "AVH+": 1}

    with open(tsv_path) as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            pid = row['participant_id']
            group = row['group']
            if group in label_map:
                labels[pid] = label_map[group]

    return labels