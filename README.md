# EXERCISE-MESURING-LAYERWISE-PERFORMANCE

## This Content was Created by Intel Edge AI for IoT Developers UDACITY Nanodegree. (Solution of the exercise and adaptation as a repository: Andrés R. Bucheli.)

Use a model with more efficient layers and see the difference in performance.

## Exercise: Measuring Layerwise Performance

In the previous section you saw the layerwise performance of the base_cnn model. In this section, we will use a model with more efficient layers and see the difference in
performance.

## Task 1
For this exercise, you will need to finish the <code>perf_counts.py</code> file on the right. This file should load the <code>sep_cnn</code> model and find the layerwise 
performance of the model.

Before running the file, make sure that you source the OpenVINO environment.

The solution uses the <code>IEPlugin</code> API to load the model. This is an alternate API to the <code>IECore</code> API that you may have seen in a previous course. You can use
either API in your solution code, but note that the <code>IEPlugin</code> API will be deprecated in a future OpenVINO version.

You can replace the <code>IEPlugin</code> code with the following:

<pre><code>
core = IECore()
net = core.load_network(network=model, device_name=args.device, num_requests=1)
</code></pre>

## Task 2
To see the performance improvements when using pooling layers, try the following tasks:

Load the <code>pool_cnn</code> model using the same script
Compare the performance of the two models

The <code>pool_cnn</code> model uses standard convolutions, whereas the <code>sep_cnn</code> model uses separable convolutions. The first convolutional layers in both these 
models have the same input and output shape. In which model does the layer execute faster?

<code>sep_cnn</code>

The <code>sep_cnn</code> model uses depthwise convolutions which require fewer FLOPs to execute. This is why it executes faster.

<pre><code>
import numpy as np
from openvino.inference_engine import IENetwork
from openvino.inference_engine import IEPlugin

import pprint
import argparse
import sys

def main(args):
    pp = pprint.PrettyPrinter(indent=4)
    model=args.model
    device=args.device
    image_path=args.image

    # Loading model
    model_weights=model+'.bin'
    model_structure=model+'.xml'
    
    # TODO: Load the model
    model=IENetwork(model_structure, model_weights)
    plugin = IEPlugin(device = device)
    
    net = plugin.load(network=model, num_requests=1)
    
    input_name=next(iter(model.inputs))

    # Reading and Preprocessing Image
    input_img=np.load(image_path)
    input_img=input_img.reshape(1, 28, 28)

    # TODO: Run Inference and print the layerwise performance
    net.requests[0].infer({input_name:input_img})
    pp.pprint(net.requests[0].get_perf_counts())

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--device', default='CPU')
    parser.add_argument('--image', default=None)
    
    args=parser.parse_args()
    sys.exit(main(args) or 0)
</code></pre>

## Run the following command to run the app: Python perf_counts.py --image image.npy --model sep_cnn/sep_cnn

## Solution of the exercise and adaptation as a Repository: Andrés R. Bucheli.
