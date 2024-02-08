import matplotlib.pyplot as plt
import os
from astropy.io.votable import parse_single_table


def plot(path=None, output_dir=os.getcwd(), subplot=False, save_plot=False):
    '''
    Parameters
    ----------
        path: str, optional
            Path to a file or a directory of files to be displayed. If there are more than 5 files in the directory, the files are saved to a subdirectory.
        output_dir: str, optional
            Directory to save the plots to. If not specified, the plots are saved to the current working directory.
        subplot: bool, optional
            If True, the plots are displayed as subplots in a single figure. Default is False.
        save_plot: bool, optional
            If True, the plots are saved to the output directory. Default is False.
    Returns
    -------
        None
    '''
    if path:
        if os.path.isfile(path):
            table = parse_single_table(path)
            item = [table]
            dl_key = os.path.basename(path)
            datalink_dict = {dl_key: item}
            extract_dl_ind(datalink_dict, dl_key, figsize=[
                           15, 5], output_dir=output_dir, save_plot=save_plot)
        elif os.path.isdir(path):
            datalink_dict = {}
            for file in os.listdir(path):
                item = []
                file_path = os.path.join(path, file)
                table = parse_single_table(file_path)
                item.append(table)
                dl_key = os.path.basename(file_path)
                datalink_dict[dl_key] = item
            extract_dl_ind(datalink_dict, figsize=[
                           15, 5], output_dir=output_dir, subplot=subplot, save_plot=save_plot)
        else:
            print(
                f"Invalid path: {path}. Please provide a valid file or directory path.")
    else:
        print("No path provided. Please provide a valid file or directory path.")
    return


def extract_dl_ind(datalink_dict, key=None, figsize=[15, 5], fontsize=12, linewidth=2, show_legend=True, show_grid=True, output_dir=None, subplot=False, save_plot=False):
    """
    Extract individual DataLink products and export them to an Astropy Table
    """
    keys = datalink_dict.keys()
    for key in keys:
        for i in datalink_dict[key]:
            if len(datalink_dict) > 5:
                dl_out = i.to_table()
                if 'wavelength' in dl_out.keys():
                    if len(dl_out) == 343:
                        title = key
                    if len(dl_out) == 2401:
                        title = key
                    plot_sampled_spec(dl_out, color='blue', title=title, fontsize=fontsize, show_legend=False,
                                      show_grid=show_grid, linewidth=linewidth, legend='', figsize=figsize, show_plot=False, save_plot=True, output_dir=output_dir)

            elif len(datalink_dict) >= 1 and len(datalink_dict) < 6:
                dl_out = i.to_table()
                if 'wavelength' in dl_out.keys():
                    if len(dl_out) == 343:
                        title = key
                    if len(dl_out) == 2401:
                        title = key
                    plot_sampled_spec(dl_out, color='blue', title=title, fontsize=fontsize, show_legend=False,
                                      show_grid=show_grid, linewidth=linewidth, legend='', figsize=figsize,
                                      show_plot=True, save_plot=False, output_dir=output_dir, subplot=subplot)
    return dl_out


def plot_sampled_spec(inp_table, color='blue', title='', fontsize=14, show_legend=True, show_grid=True, linewidth=2, legend='', figsize=[12, 4], show_plot=True, save_plot=False, output_dir=os.getcwd(), subplot=False):
    """
    RVS & XP sampled spectrum plotter. 'inp_table' MUST be an Astropy-table object.
    """
    if subplot:
        figures = []
        for table in inp_table:
            plt.figure(figsize=figsize)
            plt.plot(table['wavelength'], table['flux'],
                     '-', linewidth=linewidth, label=legend)
            figures.append(plt.gcf())

        for i, fig in enumerate(figures):
            plt.subplot(len(figures), 1, i+1)
            plt.imshow(fig)
            plt.axis('off')

    else:
        if show_plot:
            fig = plt.figure(figsize=figsize)
        xlabel = f'Wavelength [{inp_table["wavelength"].unit}]'
        ylabel = f'Flux [{inp_table["flux"].unit}]'
        plt.plot(inp_table['wavelength'], inp_table['flux'],
                 '-', linewidth=linewidth, label=legend)
        make_canvas(title=title, xlabel=xlabel, ylabel=ylabel,
                    fontsize=fontsize, show_legend=show_legend, show_grid=show_grid)
        if show_plot:
            plt.show()
        if save_plot:
            os.makedirs(output_dir, exist_ok=True)
            plt.savefig(fname=os.path.join(output_dir, title+'.png'))


def make_canvas(title='', xlabel='', ylabel='', show_grid=False, show_legend=False, fontsize=12):
    """
    Create generic canvas for plots
    """
    plt.title(title, fontsize=fontsize)
    plt.xlabel(xlabel, fontsize=fontsize)
    plt.ylabel(ylabel, fontsize=fontsize)
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    if show_grid:
        plt.grid()
    if show_legend:
        plt.legend(fontsize=fontsize*0.75)


def make_subplot(title='', xlabel='', ylabel='', show_grid=False, show_legend=False, fontsize=12):
    """
    Create generic canvas for plots
    """
    plt.title(title, fontsize=fontsize)
    plt.xlabel(xlabel, fontsize=fontsize)
    plt.ylabel(ylabel, fontsize=fontsize)
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    if show_grid:
        plt.grid()
    if show_legend:
        plt.legend(fontsize=fontsize*0.75)
