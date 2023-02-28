import torch
import matplotlib.pyplot as plt
import glob
import math

def average_smoothing(input, n_buckets):
    return torch.tensor([torch.mean(chunk) for chunk in torch.split(input, n_buckets)])


def load_data(base_directory, scaling_factor):
    base = f'{base_directory}/*_{scaling_factor}.pt'
    file_paths = glob.glob(base)
    file_paths.sort()
    test_accuracy = torch.load(
        file_paths[0], map_location=torch.device('cpu'))
    test_loss = torch.load(file_paths[1], map_location=torch.device('cpu'))
    train_accuracy = torch.load(
        file_paths[2], map_location=torch.device('cpu'))
    train_loss = torch.load(
        file_paths[3], map_location=torch.device('cpu'))
    return torch.tensor(test_accuracy), torch.tensor(test_loss), torch.tensor(train_accuracy), torch.tensor(train_loss)


def generate_plot(base_directory, output_name, title, smoothing_factor, loss_lim, val_acc_lim, train_acc_lim):
    scaling_factors = [1, 2, 4, 8, 16]
    fig, axes = plt.subplots(2, 2)
    fig.set_figwidth(9)
    fig.set_figheight(9)
    fig.suptitle(title, fontsize=14)
    for scaling_factor in reversed(scaling_factors):
        test_accuracy, test_loss, train_accuracy, train_loss = load_data(
            base_directory, scaling_factor)
        smoothning_chunks = math.ceil(len(train_loss) * smoothing_factor)
        if smoothing_factor > 0.0:
            train_loss = average_smoothing(train_loss, smoothning_chunks)
            train_accuracy = average_smoothing(
                train_accuracy, smoothning_chunks)
        test_index = torch.linspace(0, 80, len(test_accuracy))
        train_index = torch.linspace(0, 80, len(train_accuracy))
        if scaling_factor == 1:
            axes[0, 0].plot(test_index,
                            test_accuracy, '-.', label=f'SGD l={scaling_factor}')
            axes[0, 1].plot(train_index,
                            train_accuracy, '-.', label=f'SGD l={scaling_factor}')
            axes[1, 0].plot(test_index,
                            test_loss, '-.', label=f'SGD l={scaling_factor}')
            axes[1, 1].plot(train_index,
                            train_loss, '-.', label=f'SGD l={scaling_factor}')
        else:
            axes[0, 0].plot(test_index,
                            test_accuracy, label=f'l={scaling_factor}')
            axes[0, 1].plot(train_index,
                            train_accuracy, label=f'l={scaling_factor}')
            axes[1, 0].plot(test_index,
                            test_loss, label=f'l={scaling_factor}')
            axes[1, 1].plot(train_index,
                            train_loss, label=f'l={scaling_factor}')
    axes[0, 0].set_xlabel('Effective epochs')
    axes[0, 0].set_ylabel('Test accuracy')
    axes[0, 0].set_ylim(val_acc_lim)
    axes[0, 1].set_xlabel('Effective epochs')
    axes[1, 0].set_ylabel('Test loss')
    axes[1, 0].set_ylim(loss_lim)
    axes[1, 0].set_xlabel('Effective epochs')
    axes[0, 1].set_ylabel('Training accuracy')
    axes[0, 1].set_ylim(train_acc_lim)
    axes[1, 1].set_xlabel('Effective epochs')
    axes[1, 1].set_ylabel('Training loss')
    axes[1, 1].set_ylim(loss_lim)
    for i in range(2):
        for j in range(2):
            axes[i, j].legend()
    fig.tight_layout()
    plt.savefig(f'plots/{output_name}.pdf', bbox_inches='tight')
