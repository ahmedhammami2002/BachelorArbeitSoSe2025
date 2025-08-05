import torch

def split_by_classification(model, X):
    """
    Split data into positively and negatively classified samples
    """

    model.eval()
    with torch.no_grad():

        logits = model(X)
        predictions =  torch.argmax(logits, dim=1)

        pos_indices = torch.where(predictions == 1)[0].numpy()
        neg_indices = torch.where(predictions == 0)[0].numpy()

        X_positive = X[predictions == 1]
        X_negative = X[predictions == 0]


        return X_positive, X_negative, pos_indices, neg_indices


