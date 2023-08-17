#!/usr/bin/env python3

import numpy as np
import os
import sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

PhysNetdir = os.environ['PhysNet']
sys.path.append(PhysNetdir)

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import time
import argparse
import logging
logging.getLogger('tensorflow').disabled = True

from datetime import datetime
from neural_network.NeuralNetwork  import *
from neural_network.activation_fn  import *
from training.DataContainer import *
from training.DataProvider  import *

def predict(PNargs):
    tf.compat.v1.reset_default_graph()
    #define command line arguments
    parser = argparse.ArgumentParser(fromfile_prefix_chars='@')
    parser.add_argument("--restart", type=str, default=None,  help="restart training from a specific folder")
    parser.add_argument("--num_features", type=int,   help="dimensionality of feature vectors")
    parser.add_argument("--num_basis", type=int,   help="number of radial basis functions")
    parser.add_argument("--num_blocks", type=int,   help="number of interaction blocks")
    parser.add_argument("--num_residual_atomic", type=int,   help="number of residual layers for atomic refinements")
    parser.add_argument("--num_residual_interaction", type=int,   help="number of residual layers for the message phase")
    parser.add_argument("--num_residual_output", type=int,   help="number of residual layers for the output blocks")
    parser.add_argument("--cutoff", default=10.0, type=float, help="cutoff distance for short range interactions")
    parser.add_argument("--use_electrostatic", default=1, type=int,   help="use electrostatics in energy prediction (0/1)")
    parser.add_argument("--use_dispersion", default=1, type=int,   help="use dispersion in energy prediction (0/1)")
    parser.add_argument("--grimme_s6", default=None, type=float, help="grimme s6 dispersion coefficient")
    parser.add_argument("--grimme_s8", default=None, type=float, help="grimme s8 dispersion coefficient")
    parser.add_argument("--grimme_a1", default=None, type=float, help="grimme a1 dispersion coefficient")
    parser.add_argument("--grimme_a2", default=None, type=float, help="grimme a2 dispersion coefficient")
    parser.add_argument("--num_train", type=int,   help="number of training samples")
    parser.add_argument("--num_valid", type=int,   help="number of validation samples")
    parser.add_argument("--seed", default=42, type=int,   help="seed for splitting dataset into training/validation/test")
    parser.add_argument("--max_steps", type=int,   help="maximum number of training steps")
    parser.add_argument("--learning_rate", default=0.001, type=float, help="learning rate used by the optimizer")
    parser.add_argument("--max_norm", default=1000.0, type=float, help="max norm for gradient clipping")
    parser.add_argument("--ema_decay", default=0.999, type=float, help="exponential moving average decay used by the trainer")
    parser.add_argument("--keep_prob", default=1.0, type=float, help="keep probability for dropout regularization of rbf layer")
    parser.add_argument("--l2lambda", type=float, help="lambda multiplier for l2 loss (regularization)")
    parser.add_argument("--nhlambda", type=float, help="lambda multiplier for non-hierarchicality loss (regularization)")
    parser.add_argument("--decay_steps", type=int, help="decay the learning rate every N steps by decay_rate")
    parser.add_argument("--decay_rate", type=float, help="factor with which the learning rate gets multiplied by every decay_steps steps")
    parser.add_argument("--batch_size", type=int, help="batch size used per training step")
    parser.add_argument("--valid_batch_size", type=int, help="batch size used for going through validation_set")
    parser.add_argument('--force_weight',  default=52.91772105638412, type=float, help="this defines the force contribution to the loss function relative to the energy contribution (to take into account the different numerical range)")
    parser.add_argument('--charge_weight', default=14.399645351950548, type=float, help="this defines the charge contribution to the loss function relative to the energy contribution (to take into account the different numerical range)")
    parser.add_argument('--dipole_weight', default=27.211386024367243, type=float, help="this defines the dipole contribution to the loss function relative to the energy contribution (to take into account the different numerical range)")
    parser.add_argument('--summary_interval', type=int, help="write a summary every N steps")
    parser.add_argument('--validation_interval', type=int, help="check performance on validation set every N steps")
    parser.add_argument('--save_interval', type=int, help="save progress every N steps")
    parser.add_argument('--record_run_metadata', type=int, help="records metadata like memory consumption etc.")

    #if no command line arguments are present, config file is parsed
    config_file = os.path.join(PNargs.mlmodelin, 'config.txt')

    if os.path.isfile(config_file):
        args = parser.parse_args(["@"+config_file])
    else:
        args = parser.parse_args(["--help"])
    if PNargs.setname: PNargs.setname='_'+PNargs.setname
    #construct file path to directory which stores the best checkpoint files
    best_dir = os.path.join(PNargs.mlmodelin, 'best')
    testfile = 'PhysNet'+PNargs.setname+'.npz'
    #load dataset
    data = DataContainer(testfile)

    data_provider = DataProvider(data, 0, 0, 0, 0, seed=args.seed)

    #create neural network
    # NASdict = np.load(os.path.join(PNargs.mlmodelin, 'NAS.npz'))
    # tf.reset_default_graph()
    nn = NeuralNetwork(F=args.num_features,           
                    K=args.num_basis,                
                    sr_cut=args.cutoff,              
                    num_blocks=args.num_blocks, 
                    num_residual_atomic=args.num_residual_atomic,
                    num_residual_interaction=args.num_residual_interaction,
                    num_residual_output=args.num_residual_output,
                    use_electrostatic=(args.use_electrostatic==1),
                    use_dispersion=(args.use_dispersion==1),
                    s6=args.grimme_s6,
                    s8=args.grimme_s8,
                    a1=args.grimme_a1,
                    a2=args.grimme_a2,
                    # Eshift=NASdict['Eshift'],  
                    # Escale=NASdict['Escale'],   
                    activation_fn=shifted_softplus, 
                    seed=None,
                    scope="neural_network")
    #create placeholders for feeding data
    Eref      = tf.placeholder(tf.float32, shape=[None, ], name="Eref")
    Fref      = tf.placeholder(tf.float32, shape=[None,3], name="Fref") 
    Z         = tf.placeholder(tf.int32,   shape=[None, ], name="Z")     
    Dref      = tf.placeholder(tf.float32, shape=[None,3], name="Dref") 
    Qref      = tf.placeholder(tf.float32, shape=[None, ], name="Qref")   
    R         = tf.placeholder(tf.float32, shape=[None,3], name="R")      
    idx_i     = tf.placeholder(tf.int32,   shape=[None, ], name="idx_i") 
    idx_j     = tf.placeholder(tf.int32,   shape=[None, ], name="idx_j") 
    batch_seg = tf.placeholder(tf.int32,   shape=[None, ], name="batch_seg") 

    #model energy/forces
    Ea, Qa, Dij, nhloss = nn.atomic_properties(Z, R, idx_i, idx_j)
    energy, forces = nn.energy_and_forces_from_atomic_properties(Ea, Qa, Dij, Z, R, idx_i, idx_j, Qref, batch_seg)
    #total charge
    Qtot = tf.segment_sum(Qa, batch_seg)
    #dipole moment vector
    QR = tf.stack([Qa*R[:,0], Qa*R[:,1], Qa*R[:,2]],1)
    D  = tf.segment_sum(QR, batch_seg)

 

    #helper function to fill a feed dictionary
    def fill_feed_dict(data):
        feed_dict = { 
            Eref:      data["E"], 
            Fref:      data["F"],
            Z:         data["Z"], 
            Dref:      data["D"],
            Qref:      data["Q"],
            R:         data["R"],
            idx_i:     data["idx_i"],
            idx_j:     data["idx_j"],
            batch_seg: data["batch_seg"] 
        }
        return feed_dict

    #helper function to print errors

    def output_est(sess, all_data, data_count):
        with open(PNargs.yestfile,'w') as fy, open(PNargs.ygradxyzestfile,'wb') as fgrad:
            line="%d\n\n" % PNargs.natom
            for i in range(data_count):
                data = all_data[i]
                feed_dict = fill_feed_dict(data)
                E_tmp, F_tmp = sess.run([energy, forces], feed_dict=feed_dict)
                fy.write('%f\n' % E_tmp)
                fgrad.write(line.encode('utf-8'))
                np.savetxt(fgrad, -1*F_tmp, fmt='%20.12f', delimiter=" ")

    

    #create tensorflow session
    with tf.Session() as sess:
        #restore latest checkpoint
        checkpoint = tf.train.latest_checkpoint(best_dir)
        assert checkpoint is not None
        nn.restore(sess, checkpoint)

        #calculate errors on test data
        print(' ')
        # print('##### ------------ Estimating ------------ #####')
        
        start = time.time()
    
        output_est(sess, data, len(data))

        end = time.time()
        hours, rem = divmod(end-start, 3600)
        minutes, seconds = divmod(rem, 60)

        # print(' ')
        # print('--------------------------------------------------')
        # print('Total CPU time for estimating - {:0>2}:{:0>2}:{:05.2f}'.format(int(hours),int(minutes),seconds))
        # print('--------------------------------------------------')
        # print(' ')

    


