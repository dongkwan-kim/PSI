import os

import pandas as pd
import numpy as np

try:
    import seaborn as sns
    import matplotlib.pyplot as plt
except ImportError:
    pass


def plot_line_with_std(tuple_to_mean_list, tuple_to_std_list, x_label, y_label, name_label_list, x_list,
                       hue=None, size=None, style=None, palette=None,
                       row=None, col=None, hue_order=None,
                       markers=True, dashes=False,
                       height=5, aspect=1.0,
                       legend="full",
                       n=150, err_style="band",
                       x_lim=None, y_lim=None, use_xlabel=True, use_ylabel=True,
                       facet_kws=None,
                       base_path="../figs/",
                       custom_key="",
                       extension="png",
                       **relplot_kwargs):
    pd_data = {x_label: [], y_label: [], **{name_label: [] for name_label in name_label_list}}
    for name_tuple, mean_list in tuple_to_mean_list.items():
        if tuple_to_std_list is not None:
            std_list = tuple_to_std_list[name_tuple]
        else:
            std_list = [0 for _ in range(len(mean_list))]
        for x, mean, std in zip(x_list, mean_list, std_list):
            for name_label, value_of_name in zip(name_label_list, name_tuple):
                pd_data[name_label] += [value_of_name for _ in range(n)]
            pd_data[y_label] += list(np.random.normal(mean, std, n))
            pd_data[x_label] += [x for _ in range(n)]
    df = pd.DataFrame(pd_data)

    plot_info = "_".join([k for k in [hue, size, style, row, col] if k])
    full_path = os.path.join(base_path, "fig_line_{}_{}.{}".format(custom_key, plot_info, extension))

    plot = sns.relplot(x=x_label, y=y_label, kind="line",
                       row=row, col=col, hue=hue, style=style, palette=palette,
                       markers=markers, dashes=dashes,
                       height=height, aspect=aspect,
                       legend=legend, hue_order=hue_order, ci="sd", err_style=err_style,
                       facet_kws=facet_kws,
                       data=df,
                       **relplot_kwargs)
    plot.set(xlim=x_lim)
    plot.set(ylim=y_lim)
    if not use_xlabel:
        plot.set_axis_labels(x_var="")
    if not use_ylabel:
        plot.set_axis_labels(y_var="")
    plot.savefig(full_path, bbox_inches='tight')
    print("Saved at: {}".format(full_path))
    plt.clf()


if __name__ == '__main__':
    MODE = "ALL"  # FNTN, EMUser, ALL
    EXTENSION = "pdf"

    try:
        sns.set(style="whitegrid")
        sns.set_context("talk")
    except NameError:
        pass

    if MODE == "FNTN" or MODE == "ALL":
        TUPLE_TO_MEAN_LIST = {
            ("FNTN", "Intra-SGI", "$\lambda$-Intra"): [
                0.851807189, 0.8554216385, 0.8783132315, 0.874698782, 0.8771084189,
                0.8807228565, 0.8746987581, 0.879518044, 0.8662650347, 0.87469877],
            ("FNTN", "Intra/Inter-SGI", "$\lambda$-Inter"): [
                0.8578312872, 0.8855421422, 0.8975903272, 0.8903614162, 0.8891565918,
                0.8843373178, 0.8590361118, 0.8674698352, 0.8626505732, 0.8602409242],
        }
        TUPLE_TO_STD_LIST = {
            ("FNTN", "Intra-SGI", "$\lambda$-Intra"): [
                0.02196937685, 0.0321598755, 0.01776852901, 0.006599070793, 0.0131981375,
                0.00893517706, 0.0123457448, 0.004259676612, 0.007854473865, 0.01077624748],
            ("FNTN", "Intra/Inter-SGI", "$\lambda$-Inter"): [
                0.01979720785, 0.01127005298, 0.009524926568, 0.007854442088, 0.003299515193,
                0.005040120917, 0.02278032561, 0.02293907632, 0.02498364785, 0.02104126657],
        }
        NAME_LABEL_LIST = ["Dataset", "Model", "Hyperparameter"]
        X_LIST = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

        plot_line_with_std(
            tuple_to_mean_list=TUPLE_TO_MEAN_LIST,
            tuple_to_std_list=TUPLE_TO_STD_LIST,
            x_label="Value of $\lambda$",
            y_label="Test Acc.",
            name_label_list=NAME_LABEL_LIST,
            x_list=X_LIST,
            # hue="Dataset",
            style="Dataset",
            row=None,
            col="Hyperparameter",
            hue_order=None,
            markers=True, dashes=False,
            height=5, aspect=1.0,
            legend=False,
            n=150, err_style="band",
            x_lim=(0, None),
            y_lim=None, use_xlabel=True, use_ylabel=True,
            facet_kws=None,
            base_path="../figs/",
            custom_key="fntn",
            extension=EXTENSION)

    if MODE == "EMUser" or MODE == "ALL":
        TUPLE_TO_MEAN_LIST = {
            ("EM-User", "Intra-SGI", "$\lambda$-Intra"): [
                0.6382978499, 0.7489361405, 0.7744680524, 0.6978723168, 0.71914891, 0.7021276355, 0.6936169982],
            ("EM-User", "Intra/Inter-SGI", "$\lambda$-Inter"): [
                0.7234042287, 0.7574467778, 0.7234042287, 0.7276595473, 0.6893616736, 0.4893616915, 0.5276595652],
        }
        TUPLE_TO_STD_LIST = {
            ("EM-User", "Intra-SGI", "$\lambda$-Intra"): [
                0.1370648765, 0.05297828985, 0.03226755123, 0.03496101014, 0.02774128352, 0.05424488198, 0.02425904871],
            ("EM-User", "Intra/Inter-SGI", "$\lambda$-Inter"): [
                0.05211679683, 0.03226755123, 0.01504482334, 0.06626557296, 0.1255138808, 0.07056649221, 0.04092635503],
        }
        NAME_LABEL_LIST = ["Dataset", "Model", "Hyperparameter"]
        X_LIST = [0, 0.5, 1, 1.5, 2, 3, 4]

        plot_line_with_std(
            tuple_to_mean_list=TUPLE_TO_MEAN_LIST,
            tuple_to_std_list=TUPLE_TO_STD_LIST,
            x_label="Value of $\lambda$",
            y_label="Test Acc.",
            name_label_list=NAME_LABEL_LIST,
            x_list=X_LIST,
            # hue="Dataset",
            style="Dataset",
            row=None,
            col="Hyperparameter",
            hue_order=None,
            markers=True, dashes=False,
            height=5, aspect=1.0,
            legend=False,
            n=150, err_style="band",
            x_lim=(0, None),
            y_lim=None, use_xlabel=True, use_ylabel=True,
            facet_kws=None,
            base_path="../figs/",
            custom_key="emuser",
            extension=EXTENSION)