"""
Python Script that train a graph neural network for point cloud prediction using the Mixamo dataset
"""


import os
#os.environ['OPENBLAS_NUM_THREADS'] = '1'
#os.environ["CUDA_VISIBLE_DEVICES"]="9"
import sys
import io
from datetime import datetime
import argparse
import numpy as np
import time
from PIL import Image
import tensorflow as tf
from sklearn.decomposition import PCA
from datasets.Mixamo_Eval import Bodys as Dataset_Bodies_eval
from scipy.spatial import distance


#Adaptive Models
import models.AGAR as models



BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)

parser = argparse.ArgumentParser()

# Input Arguments
parser.add_argument('--data-path', default='/home/uceepdg/profile.V6/Desktop/Datasets/NPYs_Bodys', help='path')
parser.add_argument('--manual-ckpt',type=int, default=0, help='Manual restore ckpt default[False]')
parser.add_argument('--ckpt-step', type=int, default=200000, help='Manual Checkpoint step [default: 200000]')
parser.add_argument('--num-points', type=int, default=1000, help='Number of points [default: 4000]')
parser.add_argument('--num-samples', type=int, default=8, help='Number of samples [default: 4]')
parser.add_argument('--seq-length', type=int, default=12, help='Length of sequence [default: 12]')
parser.add_argument('--mode', type=str, default='basic', help='Basic model or advanced model [default: advanced]')
parser.add_argument('--unit', type=str, default='graphrnn', help='Unit. pointrnn, pointgru or pointlstm [default: pointlstm]')
parser.add_argument('--down-points1', type= int , default = 2 , help='restore-training [default:2 Downsample frames at layer 1')
parser.add_argument('--down-points2', type= int , default = 2*2, help='restore-training [default:2 Downsample frames at layer 2')
parser.add_argument('--down-points3', type= int , default = 2**2*2 , help='restore-training [default:2 Downsample frames at layer 3')

parser.add_argument('--log-dir', default='bodies', help='Log dir [default: outputs/mminst]')
parser.add_argument('--version', default='v1b', help='Model version')

print("\n EVALUATION SCRIPT \n")

args = parser.parse_args()

# SET UP DIR
summary_dir = args.log_dir
summary_dir = 'summary_dir/bodies/' + summary_dir 
summary_dir += '-bodies-%s-%s'%( args.mode, args.unit)
summary_dir += '_'+ str(args.version)

args.log_dir = 'outputs/' +args.log_dir 
args.log_dir += '-bodies-%s-%s'%(args.mode, args.unit)
args.log_dir += '_'+ str(args.version)

# Call Model
model_name = args.mode.capitalize() + 'Graph' + args.unit[5:].upper() 
#model_name = 'Advanced_plus_interpolation_GraphRNN'
print("Call model: ", model_name)
Model = getattr(models, model_name)

variable_model = True

if (variable_model == False):
  model = Model(1,
                num_points=args.num_points,
                num_samples=args.num_samples,
                seq_length=args.seq_length,
                knn=True,
                is_training=False)
if (variable_model == True):
  #For variable model
  model = Model(1,
                num_points=args.num_points,
                num_samples=args.num_samples,
                seq_length=args.seq_length,
                knn=True,
                is_training=False,
                sampled_points_down1 = args.num_points/args.down_points1,
                sampled_points_down2 = args.num_points/args.down_points2,
                sampled_points_down3 = args.num_points/args.down_points3)  
            
# Checkpoint Directory
print("args.log_dir,", args.log_dir)
checkpoint_dir = os.path.join(args.log_dir, 'checkpoints')
print("checkpoint_dir", checkpoint_dir)
checkpoint_path = os.path.join(checkpoint_dir, 'ckpt')

# Restore Checkpoint
ckpt_number = 0
checkpoint_path_automatic = tf.train.latest_checkpoint(checkpoint_dir)
ckpt_number = os.path.basename(os.path.normpath(checkpoint_path_automatic))
ckpt_number=ckpt_number[5:]
ckpt_number=int(ckpt_number)
if(args.manual_ckpt == 0):
	print("Automatic Restore")
	checkpoint_path_automatic =checkpoint_path_automatic
	log = open(os.path.join(args.log_dir, 'eval_ckp_' + str(ckpt_number) +'.log'), 'w')
if(args.manual_ckpt == 1):
	checkpoint_path = os.path.join(checkpoint_dir, 'ckpt-%d'%args.ckpt_step)
	log = open(os.path.join(args.log_dir, 'eval_ckp_' + str(args.ckpt_step) +'.log'), 'w')
	ckpt_number = args.ckpt_step
	checkpoint_path_automatic =checkpoint_path




# Test Example Folder
example_dir = os.path.join(args.log_dir, 'test-examples')
example_dir = os.path.join(example_dir, '%04d'%(ckpt_number))
if not os.path.exists(example_dir):
    os.makedirs(example_dir)

#Folder for visualizations
visu_dir = os.path.join(example_dir, 'Visualizations')
if not os.path.exists(visu_dir):
    os.makedirs(visu_dir)


test_dir = os.path.join(example_dir, 'PCA')
if not os.path.exists(test_dir):
    os.makedirs(test_dir)

#Evaluation  Log    
log = open(os.path.join(args.log_dir, 'eval_ckp_top5_' + str(ckpt_number) +'.log'), 'w')
#write input arguments
log.write("\n ========  Evaluation Log ========  \n")
log.write(":Input Arguments\n")
for var in args.__dict__:
	log.write('[%10s]\t[%10s]\n'%(str(var), str(args.__dict__[var]) ) )
log.flush()

#Load Test Dataset
test_dataset = Dataset_Bodies_eval(root=args.data_path,
                        seq_length=args.seq_length,
                        num_points=args.num_points,
                        train= False) # This will be false in evaluation

print("[Dataset Loaded] ",test_dataset )

def chamfer_distance_two_frames( P1,P2):
  
  top = 100
  cd_dist1 = distance.cdist(P1, P2, 'euclidean')
  #cd_dist1 = distance.cdist(P1, P2, 'correlation')
  cd_dist1 = np.amin( abs(cd_dist1), axis=1)
  cd_dist2 = distance.cdist(P2, P1, 'euclidean')
  #cd_dist2 = distance.cdist(P2, P1, 'correlation')
  cd_dist2 = np.amin( abs(cd_dist2), axis=1)

  CD_100 = (np.sum(cd_dist1) + np.sum(cd_dist2) ) /  (P1.shape[0] ) 
  
  # Calculate 10% of points
  top = 10
  cd_dist1_10 = np.zeros( cd_dist1.shape )
  percentil = np.percentile(cd_dist1, 100 - top)
  for i in range(0, P2.shape[0]):
    if (cd_dist1[i]  >= percentil) :
      cd_dist1_10[i] = cd_dist1[i]
  
  cd_dist2_10 = np.zeros( cd_dist2.shape )
  percentil = np.percentile(cd_dist2, 100 - top)
  for i in range(0, P2.shape[0]):
    if (cd_dist2[i]  >= percentil) :
      cd_dist2_10[i] = cd_dist2[i]    

  CD_10 = (np.sum(cd_dist1_10) + np.sum(cd_dist2_10) ) /  (P1.shape[0] *top/100 ) 

  # Calculate 10% of points
  top = 5
  cd_dist1_5 = np.zeros( cd_dist1.shape )
  percentil = np.percentile(cd_dist1, 100 - top)
  for i in range(0, P2.shape[0]):
    if (cd_dist1[i]  >= percentil) :
      cd_dist1_5[i] = cd_dist1[i]
  
  cd_dist2_5 = np.zeros( cd_dist2.shape )
  percentil = np.percentile(cd_dist2, 100 - top)
  for i in range(0, P2.shape[0]):
    if (cd_dist2[i]  >= percentil) :
      cd_dist2_5[i] = cd_dist2[i]    

  CD_5 = (np.sum(cd_dist1_5) + np.sum(cd_dist2_5) ) /  (P1.shape[0] * top/100 ) 

  return (CD_100, CD_10, CD_5)

def chamfer_distance_sequence(ground, pred):
  TCD_100 = 0
  TCD_10 = 0
  TCD_5 = 0
  for i in range(2, pred.shape[0], 1):
    (CD_100, CD_10, CD_5) = chamfer_distance_two_frames(ground[i+1], pred[i])

    TCD_100 =  TCD_100 + CD_100
    TCD_10 = TCD_10 + CD_10
    TCD_5 = TCD_5 + CD_5

  
  TCD_100 = (TCD_100/ (pred.shape[0]-2) )
  TCD_10 = (TCD_10/ (pred.shape[0]-2) )
  TCD_5 = (TCD_5/ (pred.shape[0]-2) )
  
  return (TCD_100, TCD_10, TCD_5) 




def run_test_sequence(sequence_nr):
    
    print("TEST SEQUENCE: ",sequence_nr)
    batch = test_dataset[sequence_nr]
    batch =np.array(batch)
    print("batch.shape",batch.shape)
    test_seq =batch
    test_seq =np.expand_dims(test_seq, 0)
    print("test_seq.shape: " ,test_seq.shape)
    
    feed_dict = {model.inputs: test_seq}
    out = []
    
    inputs = [
    	model.predicted_frames,
    	model.downsample_frames,
    	model.predicted_motions,
    	model.loss,
    	model.emd,
    	model.cd,
    	model.out_s_xyz0,
    	model.out_s_xyz1,
    	model.out_s_xyz2,
    	model.out_s_xyz3,
    	model.out_s_feat3
    	]

    # Run Session 
    out = sess.run( inputs, feed_dict=feed_dict)
    
    [predictions,
    downsample_frames,
    predicted_motions,
    loss,
    emd, 
    cd,
    xyz0,
    xyz1,
    xyz2,
    xyz3,
    feat3
    ] = out
    
    downsample_frames = np.array(downsample_frames)
    #print("downsample_frames.shape:",downsample_frames.shape)
    #downsample_frames = np.reshape(downsample_frames,(downsample_frames.shape[1], downsample_frames.shape[0], downsample_frames.shape[2],downsample_frames.shape[3]))
    downsample_frames = downsample_frames[0]
    print("downsample_frames.shape:",downsample_frames.shape)
    
    # Save Downnsample examples 
    npy_path= os.path.join(visu_dir, 'gdt_test_' +str(sequence_nr) )
    np.save(npy_path, downsample_frames)

    #Save Prediction
    pc_prediction = predictions[0]
    print("pc_prediction.shape:",pc_prediction.shape)
    npy_path= os.path.join(visu_dir, 'pdt_test_' +str(sequence_nr) )
    np.save(npy_path, pc_prediction)
    
    predicted_motions = np.array(predicted_motions)
    predicted_motions = predicted_motions[0]
    print("predicted_motions.shape",predicted_motions.shape)   
    npy_path= os.path.join(visu_dir, 'mt_test_' +str(sequence_nr) )
    np.save(npy_path, predicted_motions)    
    
    (TCD_100, TCD_10, TCD_5)  = chamfer_distance_sequence(downsample_frames,pc_prediction )
    cd_top5 =  TCD_5
    cd_top10 = TCD_10 
    
    #print("CD 100%, CD 10%, CD 5%:", (TCD_100, TCD_10, TCD_5) )
        
    print('[%s]\tTEST:[%10d:]\t[CD,EMD, CD5]:\t%.12f\t%.12f\t%.12f\t%.12f\n'%(str(datetime.now()), sequence_nr, cd, emd, cd_top10, cd_top5) )
    
    #Write Log
    log.write('[%s]\tTEST:[%10d:]\t[CD,EMD, CD5 ]:\t%.12f\t%.12f\t%.12f\n'%(str(datetime.now()), sequence_nr, cd, emd, cd_top5))
    log.flush()
    
    
    return (cd, emd, cd_top5)



with tf.Session() as sess:
    
    # Restore Model
    print("Restore from :",checkpoint_path_automatic)
    model.saver.restore(sess, checkpoint_path_automatic)

    
    flops = tf.profiler.profile(sess.graph, options=tf.profiler.ProfileOptionBuilder.float_operation())
    parameters = tf.profiler.profile(sess.graph, options=tf.profiler.ProfileOptionBuilder.trainable_variables_parameter())
    #print("parameters", parameters)
    #log.write ('\n flops: [%s]'% (flops))
    #log.write ('\n parameters: [%s]'% (parameters))
    print("total flops:", flops.total_float_ops)
    print("total parameters:", parameters.total_parameters)
    log.write ('\ntotal flops: {}'.format(flops.total_float_ops))
    log.write ('\ntotal parameters: {} \n\n'.format(parameters.total_parameters))
    log.flush()
    
    
    print("Restore from :",checkpoint_path_automatic)
    
    #run_test_sequence(126)
    
    t_cd= t_emd = t_cd_top5 = 0
    nr_tests = 152
    for seq in range(0,nr_tests):
    	cd, emd, cd_top5 = run_test_sequence(seq)
    	t_cd = t_cd +cd
    	t_emd = t_emd + emd
    	t_cd_top5 = t_cd_top5 + cd_top5


    print("Restore from :",checkpoint_path_automatic)
    print("Total CD: ",t_cd)
    print("Total EMD: ",t_emd)
    print("Total CD/152: ",t_cd/nr_tests)
    print("Total EMD/152: ",t_emd/nr_tests)  
    print("Total CD top 5/152: ",t_cd_top5/nr_tests)     
    log.write(" Total CD  : %f \n"%(t_cd) )
    log.write(" Total EMD : %f \n"%(t_emd) )	
    log.write(" Total CD/152 : %f \n"%(t_cd/nr_tests) )
    log.write(" Total EMD/152 : %f\n"%(t_emd/nr_tests) )
    log.write(" Total CD top 5/152 : %f\n"%(t_cd_top5/nr_tests) )
    

