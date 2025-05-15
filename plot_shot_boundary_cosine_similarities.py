import matplotlib.pyplot as plt
import numpy as np
import csv
import argparse

PLOT_FIGURE_SIZE_INCHES: tuple[float, float] = (36.0, 18.0)
PLOT_FIGURE_DPI: int = 200
PLOT_HISTOGRAM_NUMBER_OF_BINS: int = 200

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Shot Plotter Using Cosine Similarity")
    parser.add_argument('--cosine-similarities-csv-path', "-csv", required=True, type=str, help="Path to CSV file containing the cosine similarities")
    parser.add_argument('--plot-base-name', required=True, type=str ,help="Base name to give the generated plot files")
    args = parser.parse_args()
    csv_path: str = args.cosine_similarities_csv_path
    plot_base_name: str = args.plot_base_name

    shot_similarities: list[float] = list()

    with open(csv_path, "r") as file:
        reader = csv.reader(file)
        for i, row in enumerate(reader):
            assert len(row) == 3
            if i == 0:
                continue # skip header
            try:
                shot_similarities.append(float(row[2]))
            except:
                print(f"row {i} invalid format")

    plt.figure(figsize=PLOT_FIGURE_SIZE_INCHES)
    plt.plot(shot_similarities)
    plt.xticks(np.linspace(0, len(shot_similarities), 5), minor = False)
    plt.xticks(np.linspace(0, len(shot_similarities), 10), minor = True)
    plt.yticks(np.linspace(0, 1, 10), minor = False)
    plt.yticks(np.linspace(0, 1, 20), minor = True)
    plt.grid(True, "major", "y", linewidth = 2, alpha = 0.5)
    plt.grid(True, "minor", "y", linewidth = 2, alpha = 0.1)
    plt.grid(True, "major", "x", linewidth = 2, alpha = 0.5)
    plt.grid(True, "minor", "x", linewidth = 2, alpha = 0.1)
    plt.xlabel("Frame #")
    plt.ylabel("Cosine Similarity (0-1)")
    plt.title("Cosine Similarity Over the Video")
    plt.savefig(f"{plot_base_name}_line_plot.webp", format="webp", pil_kwargs={'lossless': True}, dpi=PLOT_FIGURE_DPI)

    plt.figure(figsize=PLOT_FIGURE_SIZE_INCHES)
    plt.hist(shot_similarities, bins=list(np.linspace(0, 1, PLOT_HISTOGRAM_NUMBER_OF_BINS+1)), edgecolor="black")
    plt.title("Histogram of Cosine Similarities")
    plt.xlabel("Cosine Similarity (0-1)")
    plt.ylabel("Frequency")
    plt.yscale("log")
    plt.grid(True)
    plt.savefig(f"{plot_base_name}_histogram.webp", format="webp", pil_kwargs={'lossless': True}, dpi=PLOT_FIGURE_DPI)
