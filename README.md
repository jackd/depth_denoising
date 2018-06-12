Tensorflow implementation of depth denoising models.

## Setup
1. Clone the required repositories
```
cd /path/to/parent_dir
git clone https://github.com/jackd/tf_template.git      #  project structuring
git clone https://github.com/jackd/tf_toolbox.git       #  testing/profiling
git clone https://github.com/jackd/seven_scenes.git     #  dataset
git clone https://github.com/jackd/depth_denoising.git  #  this repo
export PYTHONPATH=$PYTHONPATH:/path/to/parent_dir
```
2. See [seven scenes repo](https://github.com/jackd/seven_scenes) for instructions for getting the data.

## Running
```
cd depth_denoising/scripts/
./main.py --action=vis_inputs  # or ./vis_inputs.py
./main.py --action=test
./main.py --action=profile
./main.py --action=train
./main.py --action=vis_predictions
./main.py --action=evlauate
tensorboard --logdir=../_models
```

To specify your own network, create a `params` file in `params/my_custom_id.json` and use
```
./main.py --model_id=my_custom_id --action=...
```

## Models
* SPEN: based on papers [here](https://people.cs.umass.edu/~belanger/belanger_spen_icml.pdf) and [here](https://arxiv.org/abs/1703.05667) along with accompanying [lua code](https://github.com/davidBelanger/SPEN/) on structured energy prediction networks.
