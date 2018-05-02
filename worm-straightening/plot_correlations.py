import freeimage
import pathlib
import numpy as np
import scipy
import pickle
import collections
import time
import matplotlib.pyplot as plt
from functools import partial
from zplib.image import resample
from elegant import worm_data
from zplib.scalar_stats import mcd

plt.style.use('presentation')

def plot_scatterplots(worms, save_dir, feature_glob=['95_gfp'], min_age=-np.inf, max_age=np.inf):
    """Plot scatterplots of measurements vs. lifespan
    for a set of worms from a list of Worm objects

    Parameters:
        worms: Worms list
        save_dir: path to where you want to save the files
        min_age, max_age: beginning and end values to search the features for
    
    Returns:
    """
    features=[]
    for f in feature_glob:
        features+= [t for t in dir(worms[0].td) if not t.startswith('_') and f in t]

    lifespan = worms.get_feature('lifespan')/24
    
    for f in features:
        print(f)
        #get the measurements from elegant
        measurements = np.array(worms.get_time_range(f, min_age=min_age, max_age=max_age, match_closest=True))
        mean_list = measurements.mean(axis=1).T[1]

        #if there isn't anything to plot, then don't plot it
        if np.all(np.isnan(mean_list)):
            continue

        #plot the figures and save them
        #if we are measuring more than one day, then we need to make sure that's reflected in the plot title
        if (max_age-min_age)//24 > 1:
            plot((lifespan, mean_list), save_dir, ('lifespan', f ), days=str(min_age//24)+"-"+str(max_age//24))
        else:
            plot((lifespan, mean_list), save_dir, ('lifespan', f ), days=str(min_age//24))


def plot(measurements, save_dir, features, days="all"):
    """Function to plot things in case you want to only plot
    a few features

    Parameters:
        measurements: tuple of measurements to plot (x,y)
            In general, this is (lifespan, gfp_measurement)
        save_dir: place to save the files
        features: names of the features you are trying to plot,
            inputted in the same way as the measurments (x,y)
    
    Returns:
    """
    save_dir = pathlib.Path(save_dir)
    save_path = save_dir / (features[0]+" vs "+features[1]+" "+days+" dph.png")

    #get the regression line stuff
    pearson,spearman,yp=run_stats(measurements[0],measurements[1])

    plt.scatter(measurements[0], measurements[1])
    plt.plot(measurements[0],yp, c='gray')
    title = features[0]+" vs "+features[1]+" at "+days+" dph"
    plt.style.use('seaborn-white')
    plt.xlabel((features[0]+" (dph)"), fontdict={'size':20,'family':'calibri'})
    plt.ylabel(("Mean "+features[1]), fontdict={'size':20,'family':'calibri'})
    plt.title(title, y=1.05,fontdict={'size':26,'weight':'bold','family':'calibri'})


    if pearson[1]<.00001:
        p="p<.00001"
    else:
        p="p=" + ''+ (str)(round(pearson[1],3))
    if spearman[1]<.00001:
        spearman_p="p<.00001"
    else:
        spearman_p="p=" + '' + (str)(round(spearman[1],3))
                
    ftext='$r^{2}$ = '+(str)(round(pearson[0]**2,3))+" "+p
    gtext=r'$\rho$'+" = "+(str)(round(spearman[0],3))+" "+spearman_p
    
    plt.figtext(.15,.85,ftext,fontsize=20,ha='left')
    plt.figtext(.15,.8,gtext,fontsize=20,ha='left')
 
    plt.gcf()
    plt.savefig(save_path.as_posix())
    plt.show(block=False)
    time.sleep(1)
    plt.close()
    plt.gcf().clf

def run_stats(x_list,y_list):
    """Get the pearson, spearman, and polyfit coorelations from
    the data.
    """
    pearson=np.asarray(scipy.stats.pearsonr(x_list, y_list))
    spearman=np.asarray(scipy.stats.spearmanr(x_list, y_list))
    (m,b) = np.polyfit(x_list, y_list, 1)
    yp=np.polyval([m,b], x_list)
    
    return (pearson,spearman, yp)