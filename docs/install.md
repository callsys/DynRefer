To setup the environment of DynRefer, we use `conda` to manage our dependencies. Our developers use `CUDA 11.7` to do experiments. Run the following commands to install GenPromp:
 ```
git clone https://github.com/callsys/DynRefer
cd DynRefer
unzip dynrefer/models/model_configs.zip -d dynrefer/models/

conda create -n dynrefer python=3.8 -y
conda activate dynrefer

bash scripts/setup.sh
 ```

