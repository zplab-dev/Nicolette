import pathlib
from elegant import worm_data
import _pickle as pickle
from collections import defaultdict
import matplotlib.pyplot as plt

def name_prefix(path):
    '''When loading in multiple experiments into an elegant file
    this will make the worm names the name of the expt/worm number
    '''
    return path.parent.name+' '


def load_worms(keyword_path):
    '''Load in worms to an elegant Worms object using
    name_prefix as the name prefix
    '''
    return worm_data.read_worms(keyword_path, name_prefix=name_prefix, delimeter='\t')



#COPY PASTA STUFF TO FIX INTO NICE THINGS FOR LATER
#load in left_right_annotations
import _pickle as pickle
import numpy as np

lr_annotations = pickle.load(open('/mnt/lugia_array/data/human_masks/warped_splines/left_right_annotations.p', 'rb'))
good_splines = pickle.load((open('/mnt/lugia_array/data/human_masks/spline_tcks/good_splines_tck.p','rb')))

#make dict of all worms facing left
all_left_splines={}
for img, direction in lr_annotations.items():
    key = '/mnt/lugia_array/data/human_masks/'+img 
    if direction == 'right':
        
        ct,cc,ck=good_splines[key][0]
        wt, wc, wk=good_splines[key][1]
        #reverse center tck
        new_center_t=np.absolute(ct[::-1]-ct[-1])
        new_center_c=cc[::-1]
        new_width_t=-(wt[::-1]-wt[-1])
        new_width_c=wc[::-1]
        
        #print(new_center_t)
        #print(new_center_c)
        new_center_tck = (np.array(new_center_t), np.array(new_center_c), ck)
        new_width_tck = (np.array(new_width_t), np.array(new_width_c), wk)
        
        all_left_splines[key]=(new_center_tck, new_width_tck)
    else:
        all_left_splines[key] = good_splines[key]



#make a dict of dicts from the worms object
#{worm: {timepoint: age}}
worms = worm_data.read_worms('/mnt/lugia_array/data/human_masks/annotations/*/*.tsv', name_prefix=name_prefix, delimiter='\t')
tp_age_dict={w.name: dict(zip(w.td.timepoint, w.td.age)) for w in worms}
good_splines = pickle.load((open('/mnt/lugia_array/data/human_masks/spline_tcks/good_splines_tck.p','rb')))

#make a dict of dicts
#{worm: {timepoint: (tcks)}}
from collections import defaultdict
import pathlib
splines = defaultdict(dict)
for k, v in all_left_splines.items():
    file = pathlib.Path(k)
    worm_id = file.parent.name.rsplit(' ', 2)
    #print(worm_id)
    #worm_num = file.parent.name.rsplit('_',1)
    timepoint = file.name.split(' ')[0]
    if 'mir-63' in file.parent.name:
        name = file.parent.name
    else:
        #print(file.parent.name)
        name = worm_id[0]+" Run "+worm_id[1]+" "+worm_id[2]
    splines[name][timepoint]=v


#make a dict of dicts to incorporate everything we want
#{worm: {timepoint: ((tcks), age)}}
all_together_data = defaultdict(dict)
for worm, data in splines.items():
    for timepoint, spl in data.items():
        all_together_data[worm][timepoint]=(spl, tp_age_dict[worm][timepoint])

#HISTOGRAM OF STUFFSSSS
spline_ages = []
for timepoint in all_together_data.values():
    for age_tck in timepoint.values():
        spline_ages.append(age_tck[1])

plt.hist(spline_ages, 8)
plt.show()

#get only adult wormssss
tp_egg_age = {w.name: dict(zip(w.td.timepoint, w.td.egg_age)) for w in worms}
tck_egg = defaultdict(dict)
for worm, data in splines.items():
    for timepoint, spl in data.items():
        if tp_egg_age[worm][timepoint]>0.0:
            tck_egg[worm][timepoint]=(spl, tp_egg_age[worm][timepoint])

#Get width_tcks for adult worms
width_tcks = []
for timepoint in tck_egg.values():
    for age_tck in timepoint.values():
        if age_tck[1]>0.0:
            width_tcks.append(age_tck[0][1])

#Interpolate widths to 100 points
from zplib.curve import interpolate
width_points = []
import numpy as np


for tck in width_tcks:
    width_points.append(interpolate.spline_interpolate(tck, 100))
width_points
width_points=np.array(width_points)

#PCAAAA
from zplib import pca
mean, pcs, norm_pcs, variances, total_variance,positions, norm_positions = pca.pca_dimensionality_reduce(width_points, 0.95)
variances/total_variance
norm_pcs.shape
for std in (-1,0,1):
    plt.plot(mean+std*norm_pcs[0])
plt.show()
for std in (-1,0,1):
    plt.plot(mean+std*norm_pcs[1])
plt.show()
for std in (-1,0,1):
    plt.plot(mean+std*norm_pcs[2])
plt.show()


#From PCA Go back and forth
#Go from PCA -> real widths and back (real widths -> PCA)
#Widths -> PCA
projection = pca.pca_decompose(test_width, pcs, mean)
#PCA -> Widths
test_proj = projection[:3]
positions = pca.pca_reconstruct(test_proj, pcs[:3], mean)
#make new width spline
reverse_width_spline = interpolate.fit_nonparametric_spline(np.linspace(0,1, len(positions)), positions)


