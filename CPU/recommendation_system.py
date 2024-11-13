# Imports
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error
from torch.autograd import Variable
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader, random_split

# Loading datasets
movies_df = pd.read_csv('movie.csv')
ratings_df = pd.read_csv('rating.csv')
movie_names = movies_df.set_index('movieId')['title'].to_dict()
unique_movies = len(ratings_df.movieId.unique())
unique_users = len(ratings_df.userId.unique())

# MatrixFactorization
class MatrixFactorization(torch.nn.Module):
    def __init__(self, unique_users, unique_movies, factors=20):
        super().__init__()
        self.user_factors = torch.nn.Embedding(unique_users, factors)
        self.movie_factors = torch.nn.Embedding(unique_movies, factors)
        
        torch.nn.init.xavier_uniform_(self.user_factors.weight)
        torch.nn.init.xavier_uniform_(self.movie_factors.weight)

    def forward(self, data):
        users, movies = data[:, 0], data[:, 1]
        return (self.user_factors(users) * self.movie_factors(movies)).sum(1)

    def predict(self, user, movie):
        data = torch.tensor([[user, movie]])
        return self.forward(data)


# Pytorch
class Loader(Dataset):
    def __init__(self):
        self.ratings = ratings_df.copy()

        users = ratings_df.userId.unique()
        movies = ratings_df.movieId.unique()

        self.userid2idx = {o: i for i, o in enumerate(users)}
        self.movieid2idx = {o: i for i, o in enumerate(movies)}

        self.idx2userid = {i: o for o, i in self.userid2idx.items()}
        self.idx2movieid = {i: o for o, i in self.movieid2idx.items()}

        self.ratings.movieId = ratings_df.movieId.apply(lambda x: self.movieid2idx[x])
        self.ratings.userId = ratings_df.userId.apply(lambda x: self.userid2idx[x])

        self.x = self.ratings.drop(['rating', 'timestamp'], axis=1).values
        self.y = self.ratings['rating'].values

        self.x, self.y = torch.tensor(self.x), torch.tensor(self.y, dtype=torch.float32)

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return len(self.ratings)


# Training Setup
epochs = 256
batch_size = 128

model = MatrixFactorization(unique_users, unique_movies, factors=8)

loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

dataset = Loader()

train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_set, test_set = random_split(dataset, [train_size, test_size])

print(f"Training size: {train_size}, Test size: {test_size}")

train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

for epoch in tqdm(range(epochs)):
    model.train()  
    train_losses = []

    for x, y in train_loader:

        optimizer.zero_grad()
        outputs = model(x)
        loss = loss_fn(outputs.squeeze(), y)
        train_losses.append(loss.item())
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {np.mean(train_losses):.4f}")

# Testing
model.eval()
test_preds, test_labels = [], []

with torch.no_grad():
    for x, y in test_loader:

        outputs = model(x)
        test_preds.extend(outputs.squeeze().cpu().numpy())
        test_labels.extend(y.cpu().numpy())

#Means Squared Error
mse = mean_squared_error(test_labels, test_preds)
print(f"\nTest MSE: {mse:.4f}")

# K-means Clustering
trained_movie_embeddings = model.movie_factors.weight.data.cpu().numpy()
kmeans = KMeans(n_clusters=10, random_state=0).fit(trained_movie_embeddings)

# Display top 10 movies in each cluster
for cluster in range(10):
    print(f"\nCluster #{cluster}")
    movs = []

    for movidx in np.where(kmeans.labels_ == cluster)[0]:
        movid = dataset.idx2movieid[movidx]
        rat_count = len(ratings_df[ratings_df['movieId'] == movid])
        movs.append((movie_names[movid], rat_count))

    for mov in sorted(movs, key=lambda x: x[1], reverse=True)[:10]:
        print("\t", mov[0])
