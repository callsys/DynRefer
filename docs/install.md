To setup the environment of DynRefer, we use `conda` to manage our dependencies. Our developers use `CUDA 11.7` to do experiments. Run the following commands to install GenPromp:
 ```
conda create -n dynrefer python=3.8 -y && conda activate dynrefer

pip install --upgrade pip
pip install salesforce-lavis
pip install scikit-learn
pip install SceneGraphParser
python -m spacy download en
pip install textblob
pip install DCNv4

git clone https://github.com/callsys/DynRefer
cd DynRefer
unzip dynrefer/models/model_configs.zip
 ```

