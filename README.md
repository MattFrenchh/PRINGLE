# PRINGLE

This repository contains datasets and Jupyther notebooks associated with our work *"A toolkit for mapping cell identities in relation to neighbours reveals conserved patterning of neuromesodermal progenitor populations" French et al 2025 (in revision)*.

## Purpose

PRINGLE is a collection of Jupyther notebooks designed to enable the analysis of cell fate patterning in complex 3D tissues such as mouse, fly and chick embryos from 3D biological images. 

Patterning, is usually quantified after first defining distinct categories of cells using a gating strategy on fluorescence values, either manually or with clustering methods. However, this approach overlooks biologically meaningful graded changes in identity which occur over complex 4D trajectories. Instead, PRINGLE quantifies patterning directly, using continuous fluorescence levels within each cell in relation to the continuous levels within neighbours. PRINGLE offers:

  * The capacity to define regions of interest in a way that avoids arbitrary thresholding of individual markers
  * Quantification and visualisation of the shape and steepness of cell fate gradients using neighbour-relationships.
  * Quantification and visualisation of fine-grained patterning using neighbour-relationships.
  * Faithful 2D visualisation of 3D patterning 
  * Visualisations of averaged patterning across multiple embryos of the same developmental stage and multiple conditions


## How to use

This repository contains the datasets we have used to produce the figures of our article. These can be used out of the box to learn how PRINGLE works and how to adapt the notebooks for different datasets.

In order to get started: Clone this repository, navigate to one of the Mouse, Fly or Chick folders. Start a Jupyther lab instance and run the notebooks. NB: It is important to run each notebook in order as each notebook produces outputs used as input for the next one.




