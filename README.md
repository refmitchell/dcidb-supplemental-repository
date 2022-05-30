# dcibd-supplemental-repository [![DOI](https://zenodo.org/badge/431483160.svg)](https://zenodo.org/badge/latestdoi/431483160)

This repository contains the cue combination behavioural data and
simulation software tools which were used to produce the modelling
work for the paper "Weighted cue integration for straight-line orientation"
(Shaverdian et al., in preparation).

The name occ_sim comes from [O]ptimal [C]ue [C]ombination [Sim]ulator;
the irony is not lost on us. The repository constitues a bespoke
software suite which was designed to be easily re-configured or
re-written as our understanding changed. As such, the structure is
somewhat haphazard but should be readable. There is an included pdf
which points to the code for each of the main modelling sections with
some context or additional information.

At its core the software generates simulated beetle populations under
different cue-conflict conditions. Specifically we are interested in
the integration of wind and light cues using different candidate
strategies; this includes some standard statistically optimal
integration models and a selection of sub-optimal models.

Items marked [Legacy] or [Experimental] have been left in the code but
were not used in any of the work which was published.

If you have any questions regarding the software, feel free to contact
r.mitchell@ed.ac.uk.

## Behavioural data

Behavioural data is available under occ_sim/data. Please see
occ_sim/data/readme.txt for more information.

## Installation
### Requirements
- Anaconda (https://www.anaconda.com/distribution/#download-section)
- Matplotlib
- Numpy
- PyYAML
- Scikit Learn
- PySide2 (via conda-forge)

### Process
Clone or download the repository to your preferred location (INSTALL_DIR).

Install Anaconda (Python 3 version) using the instructions/download available
here: https://www.anaconda.com/distribution/#download-section. Once you have
installed Anaconda you need to create a virtual environment.

On Linux/MacOS, open a terminal in the same directory as the software
(INSTALL_DIR):

`$: conda create --name occ_sim`

`$: conda activate occ_sim` (to deactivate, use `$: conda deactivate`)

`$: conda install --file requirements.txt`

`$: conda install -c conda-forge pyside2`

## Usage
### Quick use
If you just want to load up the software to see what it does at a high
level then use:

`$: conda activate occ_sim`

`$: python occ_sim.py`

This will give you a GUI which allows you to configure different
experimental scenarios and see how different models respond to these
scenarios. This can also give you a sense of the variability between
runs.

### From the paper

If you would like to see the code behind the results presented, then
please see INSTALL_DIR/latex/documentation.pdf. This document contains
code references for each of the modelling stages and equations.

