import argparse
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import cycle
import os

sns.set(
    style="darkgrid",
    rc={
        "figure.figsize": (10, 6),
        "axes.titlesize": 16,
        "axes.labelsize": 14,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 12,
        "lines.linewidth": 2,
    },
)

line_styles = cycle(["-", "--", "-.", ":"])

def load_data(file_patterns, sep=","):
    """
    Load and concatenate data from multiple files while adding an 'episode' column.

    Args:
        file_patterns (list): List of file name patterns to match (e.g., ["results/my_result_conn0_ep"]).
        sep (str): Separator used in the CSV files.

    Returns:
        pd.DataFrame: Combined DataFrame with an additional 'episode' column.
    """
    df_list = []
    for pattern in file_patterns:
        for file in glob.glob(pattern + "*"):
            # Extract episode number from the filename
            episode_number = int(file.split("_ep")[-1].split(".")[0])

            # Read the CSV file
            df = pd.read_csv(file, sep=sep)

            # Add the episode column
            df["episode"] = episode_number

            # Append to the list
            df_list.append(df)

    # Concatenate all DataFrames and reset the index
    return pd.concat(df_list, ignore_index=True)


def plot_metrics_per_metric(df, metrics, xaxis="step", episodes_interval=20, ma=1, output=None):
    """
    Plot separate metrics for selected episodes over time steps.

    Args:
        df (pd.DataFrame): Input data with shape (num_episodes, num_time_steps, metrics).
        metrics (list): List of metrics to plot.
        xaxis (str): The column to use as x-axis (e.g., "step").
        episodes_interval (int): Interval for selecting episodes to visualize.
        ma (int): Moving average window for smoothing.
        output (str): If provided, saves each plot to this path with the metric name.
    """
    unique_episodes = df["episode"].unique()
    selected_episodes = unique_episodes[::episodes_interval]

    for metric in metrics:
        plt.figure(figsize=(10, 6))
        line_styles = cycle(["-", "-.", "--", ":"])
        colors = cycle(sns.color_palette("colorblind", len(selected_episodes)))

        for episode in selected_episodes:
            # Filter the DataFrame for the current episode
            episode_df = df[df["episode"] == episode]

            # Extract the x and y values directly
            x = episode_df[xaxis].values
            y = episode_df[metric].values

            # Apply moving average if specified
            if ma > 1:
                y = moving_average(y, ma)

            # Plot the metric for the current episode
            plt.plot(x, y, label=f"Ep {episode}", linestyle=next(line_styles), color=next(colors))

        # Set plot details
        plt.title(f"Metric: {metric}")
        plt.xlabel(xaxis.capitalize())
        plt.ylabel(metric.capitalize())
        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")

        # Save or show the plot
        if output:
            plt.savefig(f"{output}_{metric}.pdf", bbox_inches="tight")
        plt.show()


def plot_system_metrics(df, metrics, xaxis="step", episodes_interval=20, ma=1, output=None):
    """
    Visualize how global (system) level metrics evolve during training.

    Args:
        df (pd.DataFrame): Input data containing global metrics.
        metrics (list): List of global metrics to plot (e.g., ["system_total_waiting_time"]).
        xaxis (str): Column to use as x-axis (e.g., "step").
        episodes_interval (int): Interval for selecting episodes to visualize.
        ma (int): Moving average window for smoothing.
        output (str): If provided, saves each plot to this path . e.g. "outputs"
    """
    unique_episodes = sorted(df["episode"].unique())
    #print("unique_episodes", unique_episodes)
    selected_episodes = unique_episodes[::episodes_interval]

    for metric in metrics:
        plt.figure(figsize=(10, 6))
        line_styles = cycle(["-", "--", "-.", ":"])
        colors = cycle(sns.color_palette("colorblind", len(selected_episodes)))

        for episode in selected_episodes:
            episode_df = df[df["episode"] == episode]

            # Extract the x and y values directly
            x = episode_df[xaxis].values
            y = episode_df[metric].values

            # Apply moving average if specified
            if ma > 1:
                y = moving_average(y, ma)

            # Plot the metric for the current episode
            plt.plot(x, y, label=f"Ep {episode}", linestyle=next(line_styles), color=next(colors))

        plt.title(f"System Metric: {metric}")
        plt.xlabel(xaxis.capitalize())
        plt.ylabel(metric.replace("_", " ").capitalize())
        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")

        # Save or show the plot
        if output:
            save_path = os.path.join(output, f"system_{metric}_dcrnn.pdf")
            plt.savefig(save_path, bbox_inches="tight")
        plt.show()


def plot_agent_metrics(df, agents, metrics, xaxis="step", episodes_interval=20, ma=1, output=None):
    """
    Visualize how local (agent) level metrics evolve during training to identify bottlenecks.

    Args:
        df (pd.DataFrame): Input data containing local metrics.
        agents (list): List of agent IDs (e.g., ["A0", "A1", ..., "D3"]).
        metrics (list): List of agent-level metrics to plot (e.g., ["stopped", "accumulated_waiting_time", "average_speed"]).
        xaxis (str): Column to use as x-axis (e.g., "step").
        episodes_interval (int): Interval for selecting episodes to visualize.
        ma (int): Moving average window for smoothing.
        output (str): If provided, saves each plot to this path . e.g. "outputs"
    """
    unique_episodes = sorted(df["episode"].unique())
    selected_episodes = unique_episodes[::episodes_interval]

    for metric_suffix in metrics:
        for agent in agents:
            plt.figure(figsize=(10, 6))
            line_styles = cycle(["-", "--", "-.", ":"])
            colors = cycle(sns.color_palette("colorblind", len(agents)))

            for episode in selected_episodes:
                # Construct the metric name
                metric = f"{agent}_{metric_suffix}"

                episode_df = df[df["episode"] == episode]

                # Extract the x and y values directly
                x = episode_df[xaxis].values
                y = episode_df[metric].values

                # Apply moving average if specified
                if ma > 1:
                    y = moving_average(y, ma)

                # Plot the metric for the current agent and episode
                plt.plot(x, y, label=f"{agent} (Ep {episode})", linestyle=next(line_styles), color=next(colors))

            plt.title(f"Agent: {agent}, Metric: {metric_suffix.replace('_', ' ').capitalize()}")
            plt.xlabel(xaxis.capitalize())
            plt.ylabel(metric_suffix.replace("_", " ").capitalize())
            plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")

            # Save or show the plot
            if output:
                agent_metric_filename = f"{agent}_{metric_suffix}_dcrnn.pdf"
                save_path = os.path.join(output, agent_metric_filename)
                plt.savefig(save_path, bbox_inches="tight")
            plt.show()
def heatmap_bottlenecks(df, agents, metric_prefixes, output=None, normalize=True):
    """
    Draw heatmaps for each local metric where one axis is agents and the other is timesteps, using the final episode.

    Args:
        df (pd.DataFrame): Input data with columns for steps, agents, and metrics.
        agents (list): List of agent IDs (e.g., ["A0", "A1", "A2", ...]).
        metric_prefixes (list): List of metric prefixes to visualize (e.g., ["stopped", "accumulated_waiting_time"]).
        output (str): If provided, saves each plot to this path . e.g. "outputs"
    """
    # Extract the final episode
    final_episode = df["episode"].max()
    final_df = df[df["episode"] == final_episode]

    # Iterate through the metrics
    for prefix in metric_prefixes:
        # Create a DataFrame for the heatmap
        heatmap_data = []
        for agent in agents:
            metric_name = f"{agent}_{prefix}"
            if metric_name in final_df.columns:
                heatmap_data.append(final_df[metric_name].values)

        # Transpose the data to align with agents on one axis and timesteps on the other
        heatmap_matrix = pd.DataFrame(heatmap_data, index=agents, columns=final_df["step"].values)

        if normalize:
            heatmap_matrix = (heatmap_matrix - heatmap_matrix.min(axis=0)) / (
                        heatmap_matrix.max(axis=0) - heatmap_matrix.min(axis=0))

        # Plot the heatmap
        plt.figure(figsize=(12, 8))
        sns.heatmap(
            heatmap_matrix,
            annot=False,
            cmap="YlGnBu",
            cbar_kws={"label": f"{prefix}"},
            xticklabels=10,  # Reduce x-axis ticks to avoid clutter
            yticklabels=True,
        )
        plt.title(f"Heatmap of {prefix.capitalize()} (Final Episode)")
        plt.xlabel("Timesteps")
        plt.ylabel("Agents")

        # Save or show the plot
        if output:
            save_path = os.path.join(output, f"heatmap_bottlenecks_{prefix}_dcrnn.pdf")
            plt.savefig(save_path, bbox_inches="tight")
        plt.show()

def moving_average(data, window_size):
    """Compute moving average."""
    if window_size == 1:
        return data
    return np.convolve(data, np.ones(window_size) / window_size, mode="same")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Traffic Signal Metrics Visualization")
    parser.add_argument("-f", nargs="+", required=True, help="File patterns for metrics (e.g., 'results/my_result_conn0_ep').")
    parser.add_argument("-sep", default=",", help="CSV separator.")
    parser.add_argument("-system", nargs="+", help="List of system-level metrics to plot (e.g., 'system_total_waiting_time').")
    parser.add_argument("-agents", nargs="+", help="List of agents (e.g., 'A0', 'A1', ..., 'D3').")
    parser.add_argument("-metrics", nargs="+", help="List of agent-level metric suffixes to plot (e.g., 'stopped', 'accumulated_waiting_time', 'average_speed').")
    parser.add_argument("-ma", type=int, default=1, help="Moving average window size.")
    parser.add_argument("-episodes_interval", type=int, default=20, help="Interval of episodes to visualize.")
    parser.add_argument("-heatmap", action="store_true", help="Generate heatmaps for bottlenecks.")
    parser.add_argument("-output", type=str, default=None, help="Output file prefix for saving plots.")

    args = parser.parse_args()

    # Load data
    df = load_data(args.f, sep=args.sep)

    # Plot system-level metrics
    if args.system:
        plot_system_metrics(
            df,
            metrics=args.system,
            xaxis="step",
            episodes_interval=args.episodes_interval,
            ma=args.ma,
            output=args.output,
        )

    # Plot agent-level metrics
    #if args.agents and args.metrics:
    #    plot_agent_metrics(
    #        df,
    #        agents=args.agents,
    #        metrics=args.metrics,
    #        xaxis="step",
    #        episodes_interval=args.episodes_interval,
    #        ma=args.ma,
    #        output=args.output,
    #    )

    # Generate heatmaps for bottlenecks
    if args.heatmap and args.agents and args.metrics:
        heatmap_bottlenecks(
            df,
            agents=args.agents,
            metric_prefixes=args.metrics,
            output=args.output,
        )
