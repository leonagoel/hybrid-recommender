"""
Neural Collaborative Recommender
Uses a deep learning approach (Neural Collaborative Filtering) to discover
non-linear latent features of users and items.
"""
__all__ = ["NeuralCollaborativeRecommender"]

import logging
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from src.model.validation import validate_recommendations

logger = logging.getLogger(__name__)

class NCFDataset(Dataset):
    def __init__(self, user_indices, item_indices, ratings):
        self.user_indices = torch.tensor(user_indices, dtype=torch.long)
        self.item_indices = torch.tensor(item_indices, dtype=torch.long)
        self.ratings = torch.tensor(ratings, dtype=torch.float32)

    def __len__(self):
        return len(self.ratings)

    def __getitem__(self, idx):
        return self.user_indices[idx], self.item_indices[idx], self.ratings[idx]

class NCFNetwork(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim=32, layers=None):
        super(NCFNetwork, self).__init__()
        if layers is None:
            layers = [64, 32, 16]
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)

        mlp_layers = []
        input_size = embedding_dim * 2
        for size in layers:
            mlp_layers.append(nn.Linear(input_size, size))
            mlp_layers.append(nn.ReLU())
            mlp_layers.append(nn.Dropout(p=0.2))
            input_size = size
        
        self.mlp = nn.Sequential(*mlp_layers)
        self.prediction_layer = nn.Linear(layers[-1], 1)

    def forward(self, user_indices, item_indices):
        user_embed = self.user_embedding(user_indices)
        item_embed = self.item_embedding(item_indices)
        
        # Concatenate user and item embeddings
        vector = torch.cat([user_embed, item_embed], dim=-1)
        
        mlp_output = self.mlp(vector)
        prediction = self.prediction_layer(mlp_output)
        return prediction.squeeze()

class NeuralCollaborativeRecommender:
    def __init__(self, interaction_df, embedding_dim=32, epochs=5, batch_size=256, use_implicit=True):
        """
        interaction_df: DataFrame with columns 'user_id', 'title', 'rating'.
                        Optionally 'views' and 'purchases' for implicit feedback.
        """
        self.df = interaction_df.copy()
        
        # Map users and items to integer indices
        self.users = self.df['user_id'].astype('category')
        self.titles = self.df['title'].astype('category')

        self._user_to_idx = {u: i for i, u in enumerate(self.users.cat.categories)}
        self._title_to_idx = {t: i for i, t in enumerate(self.titles.cat.categories)}
        self.title_list = list(self.titles.cat.categories)

        n_users = len(self._user_to_idx)
        n_items = len(self._title_to_idx)

        # Build training data
        user_indices = self.users.cat.codes.values
        item_indices = self.titles.cat.codes.values
        ratings = self.df['rating'].values.astype(float)

        if use_implicit:
            alpha_implicit = 0.5
            if 'purchases' in self.df.columns:
                ratings += alpha_implicit * self.df['purchases'].fillna(0).values
            if 'views' in self.df.columns:
                ratings += (alpha_implicit * 0.5) * self.df['views'].fillna(0).values

        # Device configuration
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Fallback for very small datasets
        if n_users == 0 or n_items == 0:
            self.model = None
        else:
            self.model = NCFNetwork(
                num_users=n_users, num_items=n_items, embedding_dim=embedding_dim).to(self.device)
            self._train_model(user_indices, item_indices, ratings, epochs, batch_size)

        # Build catalog map if catalog column is present
        self._catalog_map = {}
        if 'catalog' in self.df.columns:
            self._catalog_map = self.df.groupby('title')['catalog'].first().to_dict()

    def _train_model(self, user_indices, item_indices, ratings, epochs, batch_size):
        dataset = NCFDataset(user_indices, item_indices, ratings)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=0.001, weight_decay=1e-5)
        
        self.model.train()
        for epoch in range(epochs):
            total_loss = 0
            for u, i, r in dataloader:
                u, i, r = u.to(self.device), i.to(self.device), r.to(self.device)
                
                optimizer.zero_grad()
                predictions = self.model(u, i)
                loss = criterion(predictions, r)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            logger.debug(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(dataloader):.4f}")
        
        self.model.eval()

    def recommend(self, title, top_n=10, target_catalog=None):
        """
        Item-item recommendations using cosine similarity on item embeddings.
        """
        if not isinstance(top_n, int) or top_n <= 0:
            raise ValueError("top_n must be a positive integer.")
        top_n = min(top_n, 100)

        if title not in self._title_to_idx or self.model is None:
            return []

        idx = self._title_to_idx[title]
        try:
            with torch.no_grad():
                item_embeddings = self.model.item_embedding.weight.cpu().numpy()
            
            query_vec = item_embeddings[idx]
            
            norm_q = np.linalg.norm(query_vec)
            norm_items = np.linalg.norm(item_embeddings, axis=1)
            scores = np.dot(item_embeddings, query_vec) / (norm_q * norm_items + 1e-8)
            
            sim_scores = list(enumerate(scores))
            sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        except Exception as e:
            logger.error(f"NCF recommendation similarity computation failed: {e}")
            sim_scores = []

        results = []
        seen = set()
        for i, score in sim_scores:
            t = self.title_list[i]
            if t == title or t in seen:
                continue

            if target_catalog and self._catalog_map:
                item_catalog = self._catalog_map.get(t, '')
                if str(item_catalog).lower() != str(target_catalog).lower():
                    continue

            seen.add(t)
            results.append({
                'title': t,
                'collab_score': float(score),
            })
            if len(results) >= top_n:
                break

        return validate_recommendations(
            results,
            fallback_fn=lambda top_n: self._popularity_fallback(top_n, use_collab_score=True),
            top_n=top_n,
            context="NCF",
            force_padding=False
        )

    def predict_for_user(self, user_id, top_n=10, target_catalog=None):
        """
        Predicts scores for all unseen items for a user and returns top N.
        """
        if not isinstance(top_n, int) or top_n <= 0:
            raise ValueError("top_n must be a positive integer.")
        top_n = min(top_n, 100)

        mapped_user_id = user_id
        if user_id not in self._user_to_idx:
            for key in self._user_to_idx.keys():
                if str(key) == str(user_id):
                    mapped_user_id = key
                    break

        if mapped_user_id not in self._user_to_idx or self.model is None:
            logger.info("Cold-start detected for user '%s' in NCF. Falling back.", user_id)
            recs = self._popularity_fallback(top_n)
            return validate_recommendations(
                recs,
                fallback_fn=None,
                top_n=top_n,
                context="NCF",
                force_padding=False
            )

        try:
            u_idx = self._user_to_idx[mapped_user_id]
            
            # Create pairs for all items
            num_items = len(self._title_to_idx)
            u_indices = torch.tensor([u_idx] * num_items, dtype=torch.long).to(self.device)
            i_indices = torch.arange(num_items, dtype=torch.long).to(self.device)

            with torch.no_grad():
                scores = self.model(u_indices, i_indices).cpu().numpy()

            seen_items = set(
                self.df[self.df['user_id'] == mapped_user_id]['title'].tolist()
            )

            scored = []
            for i, score in enumerate(scores):
                t = self.title_list[i]
                if t in seen_items:
                    continue

                if target_catalog and self._catalog_map:
                    item_catalog = self._catalog_map.get(t, '')
                    if str(item_catalog).lower() != str(target_catalog).lower():
                        continue

                scored.append((t, float(score)))

            scored.sort(key=lambda x: x[1], reverse=True)
            results = [{'title': t, 'predicted_score': s} for t, s in scored[:top_n]]
        except Exception as e:
            logger.error(f"NCF prediction failed: {e}")
            results = []

        return validate_recommendations(
            results,
            fallback_fn=lambda top_n: self._popularity_fallback(top_n),
            top_n=top_n,
            context="NCF",
            force_padding=False
        )

    def predict_rating(self, user_id, title):
        """Predict the rating a user would give to an item."""
        mapped_user_id = user_id
        if user_id not in self._user_to_idx:
            for key in self._user_to_idx.keys():
                if str(key) == str(user_id):
                    mapped_user_id = key
                    break

        if mapped_user_id not in self._user_to_idx or \
           title not in self._title_to_idx or self.model is None:
            return None
            
        u_idx = self._user_to_idx[mapped_user_id]
        i_idx = self._title_to_idx[title]
        
        with torch.no_grad():
            u_tensor = torch.tensor([u_idx], dtype=torch.long).to(self.device)
            i_tensor = torch.tensor([i_idx], dtype=torch.long).to(self.device)
            score = self.model(u_tensor, i_tensor).item()
            
        return float(score)
    
    def _popularity_fallback(self, top_n=10, use_collab_score=False):
        logger.info("Using popularity-based fallback for cold-start user in NCF.")
        item_counts = self.df.groupby('title')['rating'].agg(['mean', 'count']).reset_index()
    
        if 'count' in item_counts.columns and not item_counts.empty:
            top_items = item_counts.nlargest(top_n, 'count')
        elif 'mean' in item_counts.columns and not item_counts.empty:
            top_items = item_counts.nlargest(top_n, 'mean')
        else:
            return []
    
        return [
            {
                'title': row['title'],
                'collab_score' if use_collab_score else 'predicted_score': round(
                    float(row.get('mean', 0.0)), 4),
                'fallback': True
            }
            for _, row in top_items.iterrows()
        ]
