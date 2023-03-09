from datetime import datetime
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
fmt='%Y-%m-%d'
def sformat(date):
    return date.strftime(fmt)

def dformat(date):
    return datetime.strptime(date,fmt)

def heatmap_plot(scores):
    """
        Function that creates and saves a heatmap plot of all the feature contributions over time, in case a targeting
        procedure is used.
        :param scores: Pandas.DataFrame with the feature contribution scores per name of variable.
    """
    #Plot settings
    matplotlib.rcParams['mathtext.fontset'] = 'stix'
    matplotlib.rcParams['font.family'] = 'STIXGeneral'
    matplotlib.pyplot.title(r'ABC123 vs $\mathrm{ABC123}^{123}$')

    #Set contributions to 0 when feature is missing

    #Create plot
    sns.set_theme()
    fig, ax = plt.subplots(figsize=(42, 5))
    plt.tick_params(axis='both', which='major', labelsize=12)
    sns.heatmap(scores, cmap="Blues", cbar=True, ax=ax)
    cax = ax.figure.axes[-1]
    cax.tick_params(labelsize=12)
    #Save plot
    ax.figure.savefig('test.png')
    plt.close()