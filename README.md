[![GH Actions Status](https://github.com/openmm/openmm/workflows/CI/badge.svg)](https://github.com/openmm/openmm/actions?query=branch%3Amaster+workflow%3ACI)
[![Conda](https://img.shields.io/conda/v/conda-forge/openmm.svg)](https://anaconda.org/conda-forge/openmm)
[![Anaconda Cloud Badge](https://anaconda.org/conda-forge/openmm/badges/downloads.svg)](https://anaconda.org/conda-forge/openmm)

## About this modification of OpeMM

This prototype plugin, allows to perform hybrid ReaxFF/MM simulations within OpenMM. In the background it uses the PuReMD implementation of ReaxFF just like AMBER. 

Setting up a simulation is just as simple as usually with OpenMM. Additionally to a normal system setup, one has to seperate each ReaxFF atom index from the MM atom indices, and set the charge in the nonbonded forces to 0 (electrostatic interactions are calculated by ReaxFF), and call the addAtom method of the ExternalPuremdForce class. Finally, remove the unnecessary forces from the ReaxFF domain (bond, torsion, angle) and of course keep those in the MM domain and do the same with constraints. 


Later on I will provide some examples and detailed descriptions of some simulation protocols.




## OpenMM: A High Performance Molecular Dynamics Library

Introduction
------------

[OpenMM](http://openmm.org) is a toolkit for molecular simulation. It can be used either as a stand-alone application for running simulations, or as a library you call from your own code. It
provides a combination of extreme flexibility (through custom forces and integrators), openness, and high performance (especially on recent GPUs) that make it truly unique among simulation codes.  

Getting Help
------------

Need Help? Check out the [documentation](http://docs.openmm.org/) and [discussion forums](https://simtk.org/forums/viewforum.php?f=161).
