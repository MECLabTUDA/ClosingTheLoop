# Closing the loop for AI-ready Radiology
This is the repository to "Closing the loop for AI-ready Radiology". 
The models are trained on private and public rsna data in order to detect pulmonary embolism (PE).

Please cite the following paper to reference:
>Fuchs M, Gonzalez C, Frisch Y et al. Closing the loop for AI-
ready radiology. Fortschr Röntgenstr 2023; DOI 10.1055/
a-2124-1958

This project was financed by the Bundesministerium für Gesundheit(BMG) with the grant [ZMVI1–2520DAT03A]:
https://www.bundesgesundheitsministerium.de/ministerium/ressortforschung/handlungsfelder/forschungsschwerpunkte/digitale-innovation/modul-3-smarte-algorithmen-und-expertensysteme/eva-ki.html

## Installation 
### Project structure
The Project is structured into five different sections.
The first sections deals with the test, train and evaluation data split (process_input). In the second section a lung segmenter is trained. The segmenter detects the lung and saves the position in bounding boxes (lung_localization). After that, another model is trained for extracting features (seresnext50). Those features are then used to train another model in order to detect pulmonary embolism. Lastly, those results can be visualized by further extracting the raw image and the attention maps.

Each of the previously mentioned sections contains a *config.py* file. There, all necessary *base* input and output paths are noted, as well as the hyperparameters if necessary. Make sure, to adapt each necessary parameter. Each config file contains several dictionaries. The dictionary is named after the schema *filename_config*.

### Preparation
Install all necessary packages by running
```python 
pip install -r requirements.txt 
```
After that, make sure to adapt all *conf.py* files for all in- and output paths.
Note, that the first section creates the output folders for the following data

- series_dict.pickle
- image_dict.pickle

- series_list_train.pickle
- image_list_train.pickle

- series_list_valid.pickle
- image_list_valid.pickle

- series_list_test.pickle
- image_list_test.pickle

### Run
After installing the necessary packages and adapting the config files, there are several ways to run this framework. 

#### Run initial shell file
On the top folder you can find a file called *run.sh*. You can simply run it, as it starts all subsequent shell files. 
Make sure to adapt all the output paths from the shell files in the subfolders.

#### Run sub shell files
Another way of running the framework is to go into each subfolder and then run all shell files in the following order:

1. process_input (run.sh)
2. lung_localization (run.sh)
3. serexnext50 (run.sh)
4. 2nd_level (run.sh)
5. seresnext50 (run_inference.sh)

Make sure to adapt the output paths in each shell file, if necessary.
#### Run everything on your own

1. Go to *process_input* and run the following commands to split the data
    1. ```python process_input_split2.py```
    2. ```python process_input_test.py```
2. Now go to *lung_localization\split2* and start the following files to train the lung segmenter and save its outputs:
    1. ```python train.py```
    2. ```python valid.py```
    3. ```python save_bbox_train.py```
    4. ```python save_bbox_valid.py```
    5. ```python save_bbox_test.py ```
3. Step into the seresnext50 folder and run the following commands to train a feature extractor and to save the features:
    1. ```python train.py```
    2. ```python valid.py ```
    3. ```python save_valid_features.py```
    4. ```python save_train_features.py```
    5. ```python save_test_features.py ```
4. Step into *2nd_level*. There, run either (or both) the files from *1class* or *2class*
    
    1. ```python 1class/seresnext50_256_1class.py```
    2. ```python 1class/validation50_256_1class.py```
    or 
    1. ```python 2class/seresnext50_128_2class.py```
    2. ```python 2class/validation50_128_2class.py```

5.  Go back to *seresnext50* to run the inference
    1. ```python extract_raw_image.py```
    2. ```python extract_test_attention_maps.py```
