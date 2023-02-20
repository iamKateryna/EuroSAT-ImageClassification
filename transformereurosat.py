from torch.utils.data import Dataset


class TransformerEuroSAT(Dataset):
    def __init__(self, dataset, feature_extractor):
        self.dataset = dataset
        self.feature_extractor = feature_extractor

    def __len__(self):

        return len(self.dataset)

    def __getitem__(self, index):
        image = self.feature_extractor(self.dataset[index][0], return_tensors='pt')
        labels = self.dataset[index][1]

        return image['pixel_values'][0], labels
