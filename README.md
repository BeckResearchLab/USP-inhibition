<img src=https://raw.githubusercontent.com/BeckResearchLab/USP-inhibition/master/img/usp-inhibition-logo.png alt="Mountain View" width="300px" height="40px">

[![Join the chat at https://gitter.im/USP-inhibition/Lobby](https://badges.gitter.im/USP-inhibition/Lobby.svg)](https://gitter.im/USP-inhibition/Lobby?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)
[![Build Status](https://travis-ci.org/BeckResearchLab/USP-inhibition.svg?branch=master)](https://travis-ci.org/BeckResearchLab/USP-inhibition?branch=master)

## Introduction

<p align="justify">
<b> USP-inhibition </b> is a Python package for the analysis of publically available enzyme inhibition data. 
<p>In this project, we build and use quantitative structure-activity relationships (QSAR) models for the prediction of a desired interaction between enzymes and small drug molecules. The data describes the inhibition of USP1 - an enzyme essential to DNA-repair in proliferating cancer cells. Descriptors of the molecular structures of these drugs are computed to populate a working data set from the raw data in the high-throughput screen. </p>
<img src="https://raw.githubusercontent.com/BeckResearchLab/USP-inhibition/master/img/usp1_model_structure.png" align="center" alt="Modeled structure of USP1 catalytic domains using SWISS-MODEL">

Objectives: 
<ol>
<li> Engineer molecular features and build machine learning models to predict the inhibition activity of small molecules. 
<li> Use genetic algorithms to tease out optimal values of descriptors that contribute to high inhibitory action.
<li> Create an reusable open-source tool for cheminformaticians that acts as the first step to intelligent drug design prior to synthesis and testing in a lab.
</ol>

## Resources

* Dataset: view latest version on [PubChem](https://pubchem.ncbi.nlm.nih.gov/bioassay/743255) or the version used in this project on [Amazon S3](https://s3-us-west-2.amazonaws.com/pphilip-usp-inhibition/data/)
* [Previous work using this dataset](http://www.ncbi.nlm.nih.gov/pmc/articles/PMC4427583/pdf/11693_2015_Article_9162.pdf)

## Dependencies

<ul>
<li><a href="http://bioinformatics.oxfordjournals.org/content/29/8/1092.long/">
ChemoPy
</a>
<li><a href="http://openmopac.net/">
MOPAC
</a>
<li><a href="http://openbabel.org/wiki/Main_Page"> 
Open Babel
</a>
<li><a href="https://openbabel.org/docs/dev/UseTheLibrary/Python_Pybel.html">
Pybel
</a>
<li><a href="https://code.google.com/archive/p/pychem/"> 
PyChem
</a>
<li><a href="http://www.rdkit.org/"> 
RDKit
</a>
<li> Significant Python packages:
<ul>
<li> <a href="https://pythonhosted.org/nolearn/">
nolearn 
</a>
<li> <a href="http://scikit-learn.org/stable/">
scikit-learn
</a>
</ul>
</ul>

## How to use this package

Package requirements are listed [here](https://github.com/BeckResearchLab/USP-inhibition/docs/requirements.txt).

Setup instructions for this package:
- [Ubuntu](https://github.com/BeckResearchLab/USP-inhibition/docs/setup_instructions.txt)
- Windows
