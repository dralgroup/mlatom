import tensorflow as tf 
from tensorflow.python.ops.numpy_ops import np_config
np_config.enable_numpy_behavior()
import numpy as np
import sys 
sys.path.append('/export/home/chenyxx/soft/MLatom/bin')
from functions4Dtf import *

##### IG #####
# https://www.tensorflow.org/tutorials/interpretability/integrated_gradients
def interpolate_inputs(baseline, inputs, alphas):
    '''
    Generate interpolated inputs along alpha intervals from baseline to the original input
    @param baseline: starting point to perfrom interpolation
    @param input: original input to the model
    @param alphas: intervals to perform interpolation
    @return: a list of interpolated inputs
    '''
    alphas_x = alphas[:, tf.newaxis]
    baseline_x = tf.expand_dims(baseline, axis=0)
    input_x = tf.expand_dims(inputs, axis=0)
    delta = input_x - baseline_x
    inputlist = baseline_x + alphas_x * delta 
    return inputlist

def compute_gradients(inputs, fourd):
    '''
    Compute gradients for each input to the model
    @param inputs: input to the model
    @param fourd: model
    @return: gradients of the inputs (respect to output 56, see paper)
    '''
    with tf.GradientTape() as tape:
        tape.watch(inputs)
        predictions = fourd(inputs)[0][56]
    return tape.gradient(predictions, inputs)

def integral_approximation(gradients):
    '''
    Accumulate gradients
    @param gradients: gradients along alpha intervals
    @return: IG values for original inputs without scaling
    '''
    grads = (gradients[:-1] + gradients[1:]) / 2
    integrated_gradients = tf.math.reduce_mean(grads, axis=0)
    return integrated_gradients


##### generate origianl input for 4D model#####
icdict=readIC('azobenzene.ic')
azo, _ = loadXYZ(fname='azobenzene.xyz', getsp=False)
x = DescribeWithoutV(azo,icdict)
baseline = np.concatenate([x, np.zeros((1, 73))], axis=1)

azo, azo_sp = loadXYZ(fname='traj58_117200.xyz', getsp=True)
vazo, vazo_sp = loadXYZ(fname='traj58_117200.vxyz', getsp=False)
x = Describe(azo, vazo, icdict)
inputs = np.concatenate([x, [[10]]], axis=1)


##### load model #####
fourd_tc10 = tf.keras.models.load_model('./4D_model',compile=False)


##### calculate IG #####
m_steps = 100
alphas = tf.linspace(start=0.0, stop=1.0, num=m_steps+1)

inputlist = interpolate_inputs(baseline, inputs, alphas)
path_gradients = np.zeros((len(alphas), 145))
for jj in range(len(alphas)):
    eachinput = inputlist[0][jj].reshape(1, 145)
    path_gradients[jj] = compute_gradients(eachinput, fourd_tc10)
IG = integral_approximation(path_gradients)*(inputs-baseline)

