We use `conda` to manage our dependencies. Our developers use `CUDA 12.1` to do experiments. Run the following commands to setup the environment of DynRefer:
 ```
git clone https://github.com/callsys/DynRefer
cd DynRefer
unzip dynrefer/models/model_configs.zip -d dynrefer/models/

conda create -n dynrefer python=3.8 -y
conda activate dynrefer

bash scripts/setup.sh
 ```

