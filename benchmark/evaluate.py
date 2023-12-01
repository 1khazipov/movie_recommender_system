import pandas as pd
from torch.utils.data import DataLoader
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
from tabulate import tabulate
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset


class RecommenderDataset(Dataset):
    """
    dataset for the recommender system
    """
    def __init__(self, dataframe):
        self.data = dataframe.values
        self.columns = dataframe.columns

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return torch.tensor(self.data[index], dtype=torch.float32)


class EmbeddingRecommenderModel(nn.Module):
    """
    embedding model for recommender system
    args:
        user_size: number of users
        item_size: number of items
        users_features: number of features for users
        item_features: number of features for items
        embedding_size: size of the embedding
    """
    def __init__(self, user_size, item_size, users_features, item_features, embedding_size=64):
        super(EmbeddingRecommenderModel, self).__init__()
        self.user_embedding = nn.Embedding(user_size, embedding_size)
        self.item_embedding = nn.Embedding(item_size, embedding_size)
        
        self.fc1 = nn.Linear(embedding_size * 2 + item_features + users_features - 1, 64)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(64, 1)
    
    def forward(self, user_id, item_id, users_info, items_info):
        user_embed = self.user_embedding(user_id)
        item_embed = self.item_embedding(item_id)
        x = torch.cat([user_embed, item_embed, users_info, items_info], dim=1)
        x = self.relu1(self.bn1(self.fc1(x)))
        x = self.fc2(x)
        return x


def main():
    # load the test dataset
    test_dataset_path = '../benchmark/data/test_dataset.csv'
    test_data = pd.read_csv(test_dataset_path)

    user_encoder = LabelEncoder()
    item_encoder = LabelEncoder()

    test_data['user_id'] = user_encoder.fit_transform(test_data['user_id'])
    test_data['item_id'] = item_encoder.fit_transform(test_data['item_id'])
    ratings = test_data['rating'].copy()
    test_data.drop(columns=['rating'], inplace=True)
    test_data['rating'] = ratings / 5.0
    test_data['age'] /= test_data['age'].max()
    test_data['year'] /= test_data['year'].max()

    # load the trained model
    model_path = '../models/best_pytorch_model.pth'
    model = EmbeddingRecommenderModel(test_data['user_id'].nunique(), test_data['item_id'].nunique(), 23, 21)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # create a dataset for testing
    test_dataset = RecommenderDataset(test_data)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    # make predictions on the test set
    predictions = []
    targets = []

    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to("cpu")
            inputs, target = batch[:, :-1], batch[:, -1]
            user_ids = inputs[:, 0].long()
            item_ids = inputs[:, 1].long()
            item_info = inputs[:, 2:22].long()
            user_info = inputs[:, 22:].long()
            output = model(user_ids, item_ids, user_info, item_info).squeeze()
            predictions.extend(output.cpu().numpy())
            targets.extend(target.cpu().numpy())

    # calculate and print the Mean Squared Error (MSE) on the test set
    mse = mean_squared_error(targets, predictions)
    print(f'Mean Squared Error on Test Set: {mse}')

if __name__ == '__main__':
    main()
