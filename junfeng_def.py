###########################################
# Poison a model
###########################################
from models.resnet import resnet18
import torch
from data.load_data import LibriSpeechDataset, PoisonedLibriDataset, collate_fn
from torch.utils.data import Dataset, DataLoader
from torch import nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
benign_model = resnet18(classes=10).to(device)
benign_model.load_state_dict(torch.load("benign.pth"))

poisoned_model = resnet18(classes=10).to(device)
poisoned_model.load_state_dict(torch.load("poisoned.pth"))


benign_dataset_path = "original_data"
watermarked_dataset_path = "watermarked_data"

benign_dataset = LibriSpeechDataset(benign_dataset_path, validation=True)
benign_dataloader = DataLoader(benign_dataset, batch_size=10, collate_fn=collate_fn, shuffle=False)
poison_dataset = PoisonedLibriDataset(benign_dataset_path, watermarked_dataset_path, 0.5)
poison_dataloader = DataLoader(poison_dataset, batch_size=10, collate_fn=collate_fn, shuffle=False)


optimizer_poison = torch.optim.SGD(poisoned_model.parameters(), lr=0.01, nesterov=True, momentum=0.9, weight_decay=0.0005)
criterion = nn.CrossEntropyLoss()

def eva_bengin_model():
    correct_counts = 0
    for it, (audio_paths, audios, specs, mels, phases, labels) in enumerate(benign_dataloader):
        # benign sample -> benign model
        if specs.shape[0] == 10:
            audios, specs, labels = audios.to(device), specs.to(device), labels.to(device)
            logits, tuple = benign_model(specs)
            _, cls_pred = logits.max(dim=1)
            correct_count = torch.sum(cls_pred == labels.data).item()
            correct_counts += correct_count
    print ("Benign Model, Benign Sample, Acc : {}".format(correct_counts/len(benign_dataset)))

    correct_counts = 0
    for it, (audio_paths, audios, specs, mels, phases, labels) in enumerate(poison_dataloader):
        # benign sample -> benign model
        if specs.shape[0] == 10:
            audios, specs, labels = audios.to(device), specs.to(device), labels.to(device)
            logits, tuple = benign_model(specs)
            _, cls_pred = logits.max(dim=1)
            correct_count = torch.sum(cls_pred == labels.data).item()
            # print (cls_pred, labels.data, correct_count)
            correct_counts += correct_count
    print ("Benign Model, Watermarked Sample, Acc : {}".format(correct_counts/len(poison_dataset)))

# eva_bengin_model()

def eva_poison_model():
    correct_counts = 0
    for it, (audio_paths, audios, specs, mels, phases, labels) in enumerate(benign_dataloader):
        # benign sample -> watermarked model
        if specs.shape[0] == 10:
            audios, specs, labels = audios.to(device), specs.to(device), labels.to(device)
            logits, tuple = poisoned_model(specs)
            _, cls_pred = logits.max(dim=1)
            correct_count = torch.sum(cls_pred == labels.data).item()
            correct_counts += correct_count
    print ("Poisoned Model, Benign Sample, Acc : {}".format(correct_counts/len(benign_dataset)))

    correct_counts = 0
    for it, (audio_paths, audios, specs, mels, phases, labels) in enumerate(poison_dataloader):
        # watermarked sample -> benign model
        if specs.shape[0] == 10:
            audios, specs, labels = audios.to(device), specs.to(device), labels.to(device)
            logits, tuple = poisoned_model(specs)
            _, cls_pred = logits.max(dim=1)
            correct_count = torch.sum(cls_pred == labels.data).item()
            # print (cls_pred, labels.data, correct_count)
            correct_counts += correct_count
    print ("Poisoned Model, Watermarked Sample, Acc : {}".format(correct_counts/len(poison_dataset)))

eva_bengin_model()
eva_poison_model()