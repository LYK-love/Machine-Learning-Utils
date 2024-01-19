import torch

def mse(y_true, y_predicted):
    # Mean Squared Error (MSE) between x and y
    mse = torch.mean((y_true - y_predicted) ** 2)
    print("Mean Squared Error between y and \hat y:", mse.item())


def kl_divergence(true_distribution, assumed_distribution):
    '''
    Kullback-Leibler divergence (KL-divergence, relative entropy).

    Given a r.v. X with probability distribution p, suppose our model predicts, or we assume, that X's probability distributio is q,
    the KL-divergence is a kind of "distance" of our predicted `q`  while the true distribution is p.
    The lower the  KL-divergence, the closer the prediction.
    Note: p and q are multi-variance distributions if X is multidimensional..
    '''
    # Adding a small epsilon to avoid division by zero and log(0)
    epsilon = 1e-15
    true_distribution = torch.clamp(true_distribution, epsilon, 1.0)
    assumed_distribution = torch.clamp(assumed_distribution, epsilon, 1.0)

    kl_div = torch.sum(true_distribution * torch.log(true_distribution / assumed_distribution))
    return kl_div

if __name__ == "__main__":
    # Example usage
    y_true = torch.tensor([0.25, 0.25, 0.25, 0.25], dtype=torch.float32)
    y_predicted = torch.tensor([0.1, 0.2, 0.3, 0.4], dtype=torch.float32)

    kl_divergence_value = kl_divergence(y_true, y_predicted)
    print(f"KL Divergence: {kl_divergence_value.item()}")

