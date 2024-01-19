import torch
import numpy as np
import matplotlib.pyplot as plt

def calculate_cross_entropy(y_true, y_predicted):
    '''
    :param y_predicted: The predicted, often by model, distribution of data point y.
    :param y_true: The true distribution of data point y.
    :return: the cross entropy of p_predicted, given the fact that the true distribution is y_true.
    Note: y_predicted and y_true are  multi-variance distributions if X is multidimensional.
    '''
    # Ensure that p_predict and y_true have the same length
    if y_predicted.shape != y_true.shape:
        raise ValueError("Tensors p_predict and y_true must have the same shape")

    # Avoid log(0) situation
    epsilon = 1e-15
    # Clamps all elements of p_predicted in input into the range [ min, max ] where min =epsilon, and max = 1 - epsilon.
    y_predicted = torch.clamp(y_predicted, epsilon, 1 - epsilon)

    # Calculate cross entropy
    cross_entropy = -torch.sum(y_true * torch.log(y_predicted))
    return cross_entropy

def visualize_cross_entropy(save_path=None):
    '''
    Visualize the change in cross entropy,  when y_predicted=y_true, as y_predicted (or y_true, since they're equal) changes.
    :return:
    '''
    # Generate a range of predicted probabilities
    # `y_predicted` is a one-dimensional tensor of size steps whose values are evenly spaced from start (0) to end (1), inclusive.
    y_predicted = torch.linspace(0, 1, steps=100)

    # Assuming y_true is equal to y_predicted
    y_true = y_predicted

    # Calculate Cross Entropy for each predicted value
    cross_entropy_values = [calculate_cross_entropy(torch.tensor([p]), torch.tensor([p])).item() for p in y_predicted]

    # Plot
    plt.plot(y_predicted.numpy(), cross_entropy_values)
    plt.xlabel('Predicted Probability (y_predicted)')
    plt.ylabel('Cross Entropy')
    plt.title('Cross Entropy vs. Predicted Probability when y_predicted ==  y_true \n (Suppose both of them are 1-D)')

    if save_path:
        plt.savefig(save_path)
        print(f"Plot saved as {save_path}")

    plt.show()


def focal_loss(y_true, y_predicted, alpha=0.25, gamma=2.0):
    # First calculate the cross entropy loss using the helper function
    ce_loss = calculate_cross_entropy(y_true, y_predicted)

    # Calculate p_t
    p_t = y_true * y_predicted + (1 - y_true) * (1 - y_predicted)

    # Calculate the modulating factor
    modulating_factor = (1 - p_t) ** gamma

    # Apply the alpha weighting
    alpha_factor = y_true * alpha + (1 - y_true) * (1 - alpha)

    # Calculate the final Focal Loss
    focal_loss = alpha_factor * modulating_factor * ce_loss
    return focal_loss


def visualize_focal_loss_impact_of_gamma(gamma_values=[0.5, 1.0, 2.0, 5.0], save_path=None):
    # Range of predicted probabilities from 0.01 to 0.99
    y_predicted = torch.linspace(0.01, 0.99, 500)
    y_true = torch.ones_like(y_predicted)

    plt.figure(figsize=(10, 5))

    for gamma in gamma_values:
        focal_loss_values = [focal_loss(torch.tensor([1.0]), torch.tensor([p]), alpha=0.25, gamma=gamma).item() for p in
                             y_predicted]
        plt.plot(y_predicted.numpy(), focal_loss_values, label=f'gamma={gamma}')

    plt.title('Impact of gamma on Focal Loss (y_true=1)')
    plt.xlabel('Predicted Probability')
    plt.ylabel('Focal Loss')
    plt.legend()
    plt.grid()

    if save_path:
        plt.savefig(save_path)
        print(f"Plot saved as {save_path}")
    plt.show()
if __name__ == "__main__":
    # visualize_cross_entropy("outputs/Cross Entropy.png")
    # y_predicted = torch.tensor([0.8, ], dtype=torch.float32)  # Example predicted probabilities
    # y_true = torch.tensor([0.8, ], dtype=torch.float32)  # Example true labels
    #
    # cross_entropy = calculate_cross_entropy(y_predicted, y_true)
    # assert y_predicted == y_true
    # print(f"Total Cross Entropy: {cross_entropy.item()}")

    # y_true = torch.tensor([1, 0, 1, 1], dtype=torch.float32)
    # y_predicted = torch.tensor([0.9, 0.1, 0.8, 0.3], dtype=torch.float32)  # Example logits
    #
    # loss = focal_loss(y_true, y_predicted)
    # print(loss)
    visualize_focal_loss_impact_of_gamma(save_path="outputs/focal_loss.png")