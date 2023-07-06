from matplotlib import pyplot as plt

def make_histogram(values, bins, name):
    plt.close()
    plt.hist(values, bins=bins)

    # Set labels and title
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.title('Histogram')

    # Save the histogram as a PNG image
    plt.savefig(f'{name}.png')
    plt.close()