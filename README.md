# Psychedelic-ML
This repository contains projects which use machine learning to investigate the neuroscientific underpinnings of psychedelic compounds. 

**Diffusion Map Embedding for DMT:**
The first project uses a non-linear dimensionality reduction technique called diffusion map embedding to investigate how the dimethyl-tryptamine (DMT) breakdown the cortical hierarchy. In normal cognition higher-order cortical regions such as the default mode and executive networks constrain the flow of information within lower-level regions (e.g somatosensory cortices). As a result, perception is dictated heavily by the prior predictions encoded within higher-level cortical regions, as opposed to the sensory information which is gathered. The effect of DMT is to  set-off a cascade of hyperactivity within high-level cortical regions, due to their high-affinity to the 5HT2A serotonin receptor, which is particularly ubiquitous within these regions. The hyper-activity shuts down regions such as the default mode network, which allows for information to flow more freely throughout the cortex. This in turn allows for the revision of the priors which previously dominated how sensory information is interpreted. 

Diffusion map embedding (DME) was used to investigate this, as it was applied to functional connectivity data from individuals who had ingested DMT. DME reduces the dimensionality of the functional connectivity data, into axes of functional connectivity variance. The principal axes spans from higher-level cortical regions (e.g default mode network) to the lower-levels ones  (e.g. somatosensory). The analysis revealed that this axes is 'flattened' when compared to controls, as there is more integration of the functional connectivity of regions. This maps directly onto the phenomenological reports of individuals who experience aberrant cognition and perception as a result of the decomposition of the cortical hierarchy. 

Below is a visualisation of the gradients overalyed onto brain regions. It can be seen that in the DMT case there is less differentiation between brain regions, than in the placebo case, therefore demonstrating that the flow of information across regions is less constrained when psychedelics such as DMT are ingested.

[![Screenshot-2022-12-29-at-11-10-05.png](https://i.postimg.cc/SsWqPckW/Screenshot-2022-12-29-at-11-10-05.png)](https://postimg.cc/jwdGwwW5) 