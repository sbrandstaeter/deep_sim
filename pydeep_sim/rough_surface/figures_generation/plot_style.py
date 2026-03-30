from matplotlib import pyplot as plt
from matplotlib import rcParams

MY_BLUE = "#4c72b0"


def apply_plot_style():
    fontsize = 7  # pt

    # Sizes
    rcParams["axes.labelsize"] = fontsize
    rcParams["axes.titlesize"] = fontsize
    rcParams["axes.linewidth"] = 0.5
    rcParams["xtick.labelsize"] = fontsize
    rcParams["ytick.labelsize"] = fontsize
    rcParams["legend.fontsize"] = fontsize
    rcParams["font.size"] = fontsize

    # Fonts / LaTeX
    rcParams["text.usetex"] = True
    rcParams["font.family"] = "sans-serif"
    rcParams["font.sans-serif"] = "Helvetica"
    rcParams[
        "text.latex.preamble"
    ] = r"""\usepackage{siunitx}
       \sisetup{detect-all}
       \usepackage{helvet}
       \usepackage{sfmath}
       """

    # Markers
    rcParams["lines.markeredgecolor"] = "k"
    rcParams["lines.markeredgewidth"] = 0.75
    rcParams["lines.markerfacecolor"] = "darkred"
    rcParams["lines.markersize"] = 5.0

    # Lines
    rcParams["lines.linewidth"] = 1.0


def get_figure_size(textwidth_pt=507, width_fraction=0.5, aspect_ratio=1.618):
    inchesperpt = 1.0 / 72.0
    width = width_fraction * textwidth_pt * inchesperpt
    height = width / aspect_ratio
    return width, height


def style_axes(ax):
    ax.set_facecolor("#EBEBEB")
    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.set_axisbelow(True)
    ax.grid(color="white")
    ax.xaxis.set_ticks_position("none")
    ax.yaxis.set_ticks_position("none")


def make_styled_figure(
    textwidth_pt=507, width_fraction=0.5, aspect_ratio=1.618, dpi=300
):
    apply_plot_style()
    fig, ax = plt.subplots(
        figsize=get_figure_size(
            textwidth_pt=textwidth_pt,
            width_fraction=width_fraction,
            aspect_ratio=aspect_ratio,
        ),
        dpi=dpi,
    )
    style_axes(ax)
    return fig, ax
