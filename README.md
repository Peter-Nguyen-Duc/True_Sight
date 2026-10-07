# True_Sight
This repo contains the code used for generating project True Sight, which is a LLM reasoning based Anomaly detector framework. 
The framework supports general LLM agent based inserts, such that arbitrary LLM agents or even multi-agent systems can be 
used as the reasoning module for the anomaly detector module.


# Modules

This modules used for the framework is as follows

## Anomaly detector framework

A main software framework that incorporates the other modules into an anomaly detector framework


## Video handler

This module extrapolates video data, from various data sources such as:
 - .avi video data sources
 - folder paths of images in lexicographical order in accordance to their frames
 - tfrecords of image data 


## memory handler

A simple class whose purpose it is to store and retrieve video data temporarily, used by the anomaly detector.
So far it only stores the images in RAM


## True sight reasoning LLM framework

This module wraps an ollama base agentic system into a module which can be understood and utilized by the anomaly detection framework


## user interface

Not sure if i am going to develop this yet. But i think it would be cool to develop a user interface which can be used to both run 
the anomaly interfacer, but also use the anomaly interfacer for additional purposes such as video frame labeling, or alike.








