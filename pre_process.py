import numpy as np 
from nilearn import plotting
from nilearn import datasets 
from nilearn import surface 
import brainspace as brainspace  
from brainspace.utils.parcellation import reduce_by_labels    
from data import surfaces 
from nilearn.connectome import ConnectivityMeasure
from sklearn.metrics.pairwise import cosine_similarity   

"""
Prior to running this script the surface data must already be loaded into the data.py file and preferably placed in a dataloader which can be iterated over. 
This dataloader will be referred to as 'surfaces', so if its not already saved as surfaces then do so. This process should be repeated for 
both Psychedelic surface data and placebo surface data (assuming that placebo's are compared to).  

"""
 
# Fetch surface atlas
atlas = datasets.fetch_atlas_surf_destrieux()

# Remove non-cortex regions
regions = atlas['labels'].copy()
masked_regions = [b'Medial_wall', b'Unknown']
masked_labels = [regions.index(r) for r in masked_regions]
for r in masked_regions:
    regions.remove(r)

# Build Destrieux parcellation and mask
labeling = np.concatenate([atlas['map_left'], atlas['map_right']])
mask = ~np.isin(labeling, masked_labels)

# Distinct labels for left and right hemispheres
lab_lh = atlas['map_left']
labeling[lab_lh.size:] += lab_lh.max() + 1 

for surface in surfaces: 
    surface = reduce_by_labels(surface[mask], labeling[mask], axis=1, red_op='mean') 

#this dictionary will store the functional co matrices for the 
correlation_matrices = dict()

correlation_measure = ConnectivityMeasure(kind='correlation')

for idx in range(len(surfaces)):
    correlation_matrices[f'correlation_matrix{idx}'] = correlation_measure.fit_transform([surfaces[idx]])[0]  

#Plot the functional connectivity matrix 

# Reduce matrix size, only for visualization purposes
mat_mask = np.where(np.std(correlation_matrices['correlation_matrix1'], axis=1) > 0.2)[0]
c = correlation_matrices['correlation_matrix1'],[mat_mask][:, mat_mask]

# Create corresponding region names
regions_list = ['%s_%s' % (h, r.decode()) for h in ['L', 'R'] for r in regions]
masked_regions = [regions_list[i] for i in mat_mask]


corr_plot = plotting.plot_matrix(c, figure=(15, 15), labels=masked_regions,
                                 vmax=0.8, vmin=-0.8, reorder=True)  

#Z transform the matrices 
for idx in range(len(surfaces)):
    correlation_matrices[f'correlation_matrix{idx}'] = np.arctanh(correlation_matrices[f'correlation_matrix{idx}'])

#make the matrices sparse 
for matrix in correlation_matrices.keys():
    for row in correlation_matrices[matrix]: 
        for i in row:
            if i < np.min(np.percentile(row, 90)):
                row[row == i] = 0                                 

cosineSimilarity = dict() 

for idx in range(len(surfaces)):
    cosineSimilarity[f'Similarity{idx}'] = cosine_similarity(correlation_matrices[f'correlation_matrix{idx}'] )  


