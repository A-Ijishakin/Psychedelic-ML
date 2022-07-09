import numpy as np 
from pre_process import cosineSimilarity, mask, labeling
from brainspace.gradient import GradientMaps
from brainspace.datasets import load_fsa5
from brainspace.plotting import plot_hemispheres
from brainspace.utils.parcellation import map_to_labels

GM = GradientMaps(n_components=2, random_state=0)

Gradients = dict()

for idx, similarity in enumerate(cosineSimilarity):
    Gradients[f'Gradient{idx}'] = GM.fit(similarity) 

#visualise the gradients 

# Map gradients to original parcels
grad = [None] * 2
for i, g in enumerate(GM.gradients_.T):
    grad[i] = map_to_labels(g, labeling, mask=mask, fill=np.nan)


# Load fsaverage5 surfaces
surf_lh, surf_rh = load_fsa5()

# sphinx_gallery_thumbnail_number = 2
plot_hemispheres(surf_lh, surf_rh, array_name=grad, size=(1200, 400), cmap='viridis_r',
                 color_bar=True, label_text=['Grad1', 'Grad2'], zoom=1.5)    

#Procrustes Align the Gradients 
Procrustes = GradientMaps(n_components = 2, kernel='normalized_angle', alignment='procrustes')  

GradAlign = Procrustes.fit([gradient for gradient in [cosineSimilarity[f'Similarity{idx}'] for idx in range(len(cosineSimilarity.keys()))]]) 
