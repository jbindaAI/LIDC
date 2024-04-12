# Module containing some useful plotting functions.

import matplotlib.pyplot as plt


def plot_hists(y_train, y_val, y_test):
    fig, axs = plt.subplots(1, 3, figsize=(10, 5))
    _, _, bars = axs[0].hist(y_train, bins=3)
    axs[0].bar_label(bars)
    axs[0].set_title("train set")
    axs[0].set_ylabel("counts")
        
    _, _, bars = axs[1].hist(y_val, bins=3)
    axs[1].bar_label(bars)
    axs[1].set_title("val set")
    axs[1].set_ylabel("counts")

    _, _, bars = axs[2].hist(y_test, bins=3)
    axs[2].bar_label(bars)
    axs[2].set_title("test set")
    axs[2].set_ylabel("counts")

    fig.tight_layout()
    fig.subplots_adjust(top=0.85)
    fig.suptitle('Distribution of classes', size=16)
    return None