import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from openTSNE import TSNE

def plot_adj(adj_mx, save_path):
    plt.ioff()
    plt.figure(figsize=(16, 16))
    plt.imshow(adj_mx, cmap='plasma', interpolation='nearest')
    bar = plt.colorbar()
    bar.ax.tick_params(labelsize=25)
    plt.title('Adaptive Adjacency Matrix Heatmap', fontsize=25)
    plt.xticks(fontsize=25)
    plt.yticks(fontsize=25)
    plt.savefig(save_path)
    plt.close()

def plot_edge_weight(adj_mx, save_path):
    plt.ioff()
    adj_mx = adj_mx.flatten()
    zero_weight_count = np.count_nonzero(adj_mx == 0)
    plt.figure(figsize=(16, 16))
    plt.hist(adj_mx, bins=10, color='skyblue', edgecolor='black')
    plt.xlabel('Edge Weight', fontsize=25)
    plt.ylabel('Frequency', fontsize=25)
    plt.title('Weight Distribution', fontsize=25)
    plt.text(0.05, 0.9, f'Zero Weights Count: {zero_weight_count}', transform=plt.gca().transAxes, fontsize=20)
    plt.xticks(fontsize=25)
    plt.yticks(fontsize=25)
    plt.savefig(save_path)
    plt.close()

def plot_flow(flow, save_path):
    plt.ioff()
    x = np.arange(0, len(flow), step=1)
    plt.figure(figsize=(16, 4))  # 设置图形大小
    plt.plot(x, flow)  # 绘制波形图，设置颜色为蓝色，线型为实线
    plt.xlabel('Time')  # 设置x轴标签
    plt.ylabel('Traffic Flow')  # 设置y轴标签
    plt.grid(True)  # 添加网格线
    plt.savefig(save_path)
    plt.close()

def plot_spatial_embedding(embedding, save_path):
    num_nodes = embedding.shape[0]
    tsne = TSNE(
        n_components=2,
        perplexity=15,
        metric="cosine",
        n_jobs=8,
        n_iter=1500,
        learning_rate=0.3,
        neighbors='exact',
        random_state=42,
        verbose=True,
    )
    embedding_train = tsne.fit(embedding)
    plot_tsne(x=embedding_train, y=[1 for _ in range(num_nodes)], draw_legend=False, save_path=save_path)

def plot_tsne(
        x,
        y,
        ax=None,
        title=None,
        draw_legend=True,
        draw_centers=False,
        draw_cluster_labels=False,
        colors=None,
        legend_kwargs=None,
        label_order=None,
        save_path=None,
        **kwargs
):
    plt.ioff()

    if ax is None:
        _, ax = plt.subplots(figsize=(16, 16))

    if title is not None:
        ax.set_title(title)

    plot_params = {"alpha": kwargs.get("alpha", 0.8), "s": kwargs.get("s", 60)}

    # Create main plot
    if label_order is not None:
        assert all(np.isin(np.unique(y), label_order))
        classes = [l for l in label_order if l in np.unique(y)]
    else:
        classes = np.unique(y)
    if colors is None:
        default_colors = matplotlib.rcParams["axes.prop_cycle"]
        colors = {k: v["color"] for k, v in zip(classes, default_colors())}

    point_colors = list(map(colors.get, y))

    ax.scatter(x[:, 0], x[:, 1], c=point_colors, rasterized=True, **plot_params)

    # Plot mediods
    if draw_centers:
        centers = []
        for yi in classes:
            mask = yi == y
            centers.append(np.median(x[mask, :2], axis=0))
        centers = np.array(centers)

        center_colors = list(map(colors.get, classes))
        ax.scatter(
            centers[:, 0], centers[:, 1], c=center_colors, s=80, alpha=1, edgecolor="k"
        )

        # Draw mediod labels
        if draw_cluster_labels:
            for idx, label in enumerate(classes):
                ax.text(
                    centers[idx, 0],
                    centers[idx, 1] + 2.2,
                    label,
                    fontsize=kwargs.get("fontsize", 6),
                    horizontalalignment="center",
                )

    # Hide ticks and axis
    ax.set_xticks([]), ax.set_yticks([]), ax.axis("off")

    if draw_legend:
        legend_handles = [
            matplotlib.lines.Line2D(
                [],
                [],
                marker="s",
                color="w",
                markerfacecolor=colors[yi],
                ms=10,
                alpha=1,
                linewidth=0,
                label=yi,
                markeredgecolor="k",
            )
            for yi in classes
        ]
        # legend_kwargs_ = dict(loc="center left", bbox_to_anchor=(1, 0.5), frameon=False, fontsize=12)
        # 'best', 'upper right', 'upper left', 'lower left', 'lower right', 'right', 'center left', 'center right', 'lower center', 'upper center', 'center'
        legend_kwargs_ = dict(loc="lower center", bbox_to_anchor=(0.5, -0.1), fontsize=20, ncol=4)
        if legend_kwargs is not None:
            legend_kwargs_.update(legend_kwargs)
        ax.legend(handles=legend_handles, **legend_kwargs_)

    plt.savefig(save_path)
    plt.close()