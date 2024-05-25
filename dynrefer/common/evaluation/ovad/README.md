# ovad-api
API for Open-vocabulary Attribute Detection [project page](https://ovad-benchmark.github.io/)
This README describes the dataset and associated benchmark. By the date of 11.2022.

# Dataset

The dataset is compose of a single file ovad900_train.json which contains a dictionary with the following data:

	o info
    {
        "description": "OVAD 2022 Dataset ovad",
        "url": "https://ovad-benchmark.github.io/", 
        "version": "1.0", 
        "year": 2022, 
        "contributor": "OVAD Consortium", 
        "date_created": "2022-11-01"
        "license": {
            "url": "https://creativecommons.org/licenses/by-nc-sa/4.0/",
            "id": 0,
            "name": "Commons Attribution-NonCommercial-ShareAlike 4.0 International"
        }
    }

	
	o images
	List of images on the dataset.
    [
        {
            "id": 69356, % id of the image corresponding to coco 2017 dataset
            "width": 640, % dimensions for the bounding box annotations
            "height": 505, 
            "file_name": "000000069356.jpg", % name of the image file
            "set": "train300", % set for validation or training the attributes 
            "license": 1, % licence of the image
            "date_captured": "2013-11-19 18:11:52"
        }
        ...
    ]
	
    o categories
    List of 80 object categories reference with the ms-coco ids
    [
        {
            "supercategory": "person",
            "isthing": 1,
            "id": 1,
            "name": "person",
            "ov_set": "base"
        },
        ...
    ] 

	o attributes
    List of attributes 117
    [
        {
            "id": 0,
            "name": "cleanliness:clean/neat",
            "type": "cleanliness",
            "parent_type": "cleanliness",
            "is_has_att": "is",
            "freq_set": "head"
        },
        ...
    ]

    o licenses
    [
        {
            'url': 'http://creativecommons.org/licenses/by-nc-sa/2.0/', 
            'id': 1, 
            'name': 'Attribution-NonCommercial-ShareAlike License'
        }
        ...
    ]

    o annotations
    List of object and attribute annotations 
    [
        {
            'id': 2309, % id, index of the object, 1 to 14299
            'image_id': 69356, % image of the object, corresponding to MSCOCO ids  
            'bbox': [556.67, 159.5, 20.53, 41.09], % bounding box coordinates [XYWH] 
            'area': 843.578857421875, % area in pixels 
            'iscrowd': 0, 
            'category_id': 0, % object id from 0 to 79
            'att_vec': [0, 0, 0, 0, 1, 0, -1, -1, ...] 
            % attribute annotation: list of 121 integers
            % 1 = positive attribute
            % 0 = negative attribute
            % -1 = ignore attribute
        }
        ...
    ]

In addition to this API, please download the MS-COCO validation 2017 images [MS COCO](http://cocodataset.org/).

# Evaluation

For the evaluation we consider two main files: 

1. ovad2000.json
    where the attributes are defined, the hierarchy, the synonyms, the types, and ids

2. attribute_evaluator.py
    evaluation code to compare performance of methods.
    follow the main function example to evaluate. 

# Cite

If you find this dataset and benchmark useful please cite 

@article{Bravo_2022_ovad,
    author = {Maria A. Bravo and Sudhanshu Mittal and Simon Ging and Thomas Brox},
    title = {Open-vocabulary Attribute Detection},
    journal = {arXiv preprint arXiv:2211.12914},
    url = {https://arxiv.org/abs/2211.12914},
    doi = {10.48550/ARXIV.2211.12914},
    year = {2022}
}
 
