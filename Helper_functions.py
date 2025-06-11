import torch

def split_by_classification(model, X, device='cpu'):
    """
    Split data into positively and negatively classified samples
    """

    model.eval()
    with torch.no_grad():
        X = X.to(device)
        model = model.to(device)

        # Get predictions
        logits = model(X)
        predictions =  torch.argmax(logits, dim=1)

        # Convert boolean masks to indices
        pos_indices = torch.where(predictions == 1)[0].cpu().numpy()
        neg_indices = torch.where(predictions == 0)[0].cpu().numpy()

        # Split the data
        X_positive = X[predictions == 1]
        X_negative = X[predictions == 0]

        # Move back to CPU if needed
        X_positive = X_positive.cpu()
        X_negative = X_negative.cpu()

        return X_positive, X_negative, pos_indices, neg_indices


