import os
import sys
import numpy as np
import tensorflow as tf
import math

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'modules/tf_ops/sampling'))
sys.path.append(os.path.join(ROOT_DIR,'modules/tf_ops/grouping'))
sys.path.append(os.path.join(ROOT_DIR,'modules/tf_ops/3d_interpolation'))
sys.path.append(os.path.join(ROOT_DIR, 'modules/dgcnn_utils'))


from tf_sampling import farthest_point_sample, gather_point
from tf_grouping import query_ball_point, group_point, knn_point, knn_feat
from tf_interpolate import three_nn, three_interpolate
import tf_util


""" 
================================================== 
         Original Graph-RNN cell                        
================================================== 
"""

class GraphRNNCell(object):
    def __init__(self,
                 radius,
                 nsample,
                 out_channels,
                 knn=False,
                 pooling='max',
                 activation= None):

        self.radius = radius
        self.nsample = nsample
        self.out_channels = out_channels
        self.knn = knn
        self.pooling = pooling
        self.activation = activation

    def init_state(self, inputs, state_initializer=tf.zeros_initializer(), dtype=tf.float32):
        """Helper function to create an initial state given inputs.
        Args:
            inputs: tube of (P, X). the first dimension P or X being batch_size
            state_initializer: Initializer(shape, dtype) for state Tensor.
            dtype: Optional dtype, needed when inputs is None.
        Returns:
            A tube of tensors representing the initial states.
        """
        # Handle both the dynamic shape as well as the inferred shape.
        P, C, F, X, T = inputs

        # inferred_batch_size = tf.shape(P)[0]
        inferred_batch_size = P.get_shape().with_rank_at_least(1)[0]
        inferred_npoints = P.get_shape().with_rank_at_least(1)[1]
        inferred_xyz_dimensions = P.get_shape().with_rank_at_least(1)[2]
        #inferred_feature_dimensions = 128 # ASSUMPTION
        
        P = state_initializer([inferred_batch_size, inferred_npoints, inferred_xyz_dimensions], dtype=P.dtype)
        #C = state_initializer([inferred_batch_size, inferred_npoints, inferred_xyz_dimensions], dtype=dtype)
        C = None
        #F = state_initializer([inferred_batch_size, inferred_npoints, inferred_xyz_dimensions], dtype=dtype)
        S = state_initializer([inferred_batch_size, inferred_npoints, self.out_channels], dtype=dtype)
        extra = None        

        return (P, C, F, S, T, extra)

    def __call__(self, inputs, states):
        if states is None:
            states = self.init_state(inputs)

        P1, C1, F1, X1, T1= inputs
        P2, C2, F2, S2, T2, extra = states
        
        radius = self.radius
        nsample = self.nsample
        out_channels = self.out_channels
        knn = self.knn
        pooling = self.pooling
        activation = self.activation
        
        # 1 - Create adjacent matrix on feature space F1
        P1_adj_matrix = tf_util.pairwise_distance(F1)
        P1_nn_idx = tf_util.knn(P1_adj_matrix, k= nsample)
        
        # look at neighborhoodbood in P2
       	P2_adj_matrix = tf_util.pairwise_distance_2point_cloud(F2, F1)
       	P2_nn_idx = tf_util.knn(P2_adj_matrix, k= nsample)
        
        if (knn == False) : # DO A BALL QUERY
        	print("\nBALL QUERY NOT IMPLEMENTED")
        	"""
        	idx, cnt = query_ball_point(radius, nsample, P1, P1)
        	cnt = tf.tile(tf.expand_dims(cnt, -1), [1, 1, nsample])
        	P1_nn_idx = tf.where(cnt > (nsample-1), idx, P1_nn_idx)
        	
        	idx, cnt = query_ball_point(radius, nsample, P2, P1)
        	cnt = tf.tile(tf.expand_dims(cnt, -1), [1, 1, nsample])
        	P2_nn_idx = tf.where(cnt > (nsample-1), idx, P2_nn_idx )
        	"""

        # 2.1 Group P1 points
        P1_grouped = group_point(P1, P1_nn_idx)                      
        # 2.3 Group P color
        #if (C1 is not None):
        	#C1_grouped = group_point(C1, P1_nn_idx)                       # batch_size, npoint, nsample, out_channels
        # 2.4 Group P feat
        F1_grouped = group_point(F1, P1_nn_idx)                       # batch_size, npoint, nsample, out_channels
        # 2.4 Group P time
        T1_grouped = group_point(T1, P1_nn_idx)                       # batch_size, npoint, nsample, out_channels
        # 2.2 Group P1 states
        if (X1 is not None):
        	S1_grouped = group_point(X1, P1_nn_idx)  
        
        # 2.1 Group P2 points
        P2_grouped = group_point(P2, P2_nn_idx)                      
        # 2.3 Group P color
        #C2_grouped = group_point(C2, P2_nn_idx)                       # batch_size, npoint, nsample, out_channels# 2.4 Group P feat
        F2_grouped = group_point(F2, P2_nn_idx)                       # batch_size, npoint, nsample, out_channels
        #2.4 Group S2 states
        S2_grouped = group_point(S2, P2_nn_idx)   
        # 2.4 Group P2 time
        T2_grouped = group_point(T2, P2_nn_idx)                       # batch_size, npoint, nsample, out_channels
 
        ##  Neighborhood P1"
        # 3. Calculate displacements
        P1_expanded = tf.expand_dims(P1, 2)                     # batch_size, npoint, 1,       3
        displacement = P1_grouped - P1_expanded                 # batch_size, npoint, nsample, 3
        #3.1 Calculate color displacements
        #C1_expanded = tf.expand_dims(C1, 2)                     # batch_size, npoint, 1,       3
        #displacement_color = C1_grouped - C1_expanded           # batch_size, npoint, nsample, 3
        #3.1 Calculate feature displacements
        F1_expanded = tf.expand_dims(F1, 2)                     # batch_size, npoint, 1,       3
        displacement_feat = F1_grouped - F1_expanded           # batch_size, npoint, nsample, 3
        #3.1 Calculate time displacements
        T1_expanded = tf.expand_dims(T1, 2)                     # batch_size, npoint, 1,       3
        displacement_time = T1_grouped - T1_expanded           # batch_size, npoint, nsample, 3
              
        ##  Neighborhood P2 "
        # 3. Calculate displacements
        P2_expanded = tf.expand_dims(P2, 2)                     # batch_size, npoint, 1,       3
        displacement_2 = P2_grouped - P1_expanded                 # batch_size, npoint, nsample, 3
        #3.1 Calculate color displacements
        #C2_expanded = tf.expand_dims(C2, 2)                     # batch_size, npoint, 1,       3
        #displacement_color_2 = C2_grouped - C1_expanded           # batch_size, npoint, nsample, 3
        #3.1 Calculate feature displacements
        F2_expanded = tf.expand_dims(F2, 2)                     # batch_size, npoint, 1,       3
        displacement_feat_2 = F2_grouped - F1_expanded           # batch_size, npoint, nsample, 3        
        #3.1 Calculate time displacements
        T2_expanded = tf.expand_dims(T2, 2)                     # batch_size, npoint, 1,       3
        displacement_time_2 = T2_grouped - T1_expanded           # batch_size, npoint, nsample, 3

        # 4. Concatenate X1, S2 and displacement
        if X1 is not None:
        	#print("Concatenation (t) = [F_i | S_ij | displacement_Pij | displacement_Fij| displacement_T] " )
        	X1_expanded = tf.tile(tf.expand_dims(X1, 2), [1, 1, nsample, 1])  
        	F1_expanded = tf.tile(tf.expand_dims(F1, 2), [1, 1, nsample, 1])                
        	concatenation = tf.concat([X1_expanded, S1_grouped], axis=3)         
        	concatenation = tf.concat([concatenation, displacement ,displacement_feat, displacement_time], axis=3)
        	concatenation_2 = tf.concat([X1_expanded, S2_grouped], axis=3)
        	concatenation_2 = tf.concat([concatenation_2, displacement_2,displacement_feat_2, displacement_time_2], axis=3) 
        	             
        else:
        	#print("Concatenation (t) = [displacement_Pij | displacement_Fij| displacement_T] " )
        	F1_expanded = tf.tile(tf.expand_dims(F1, 2), [1, 1, nsample, 1])                        
        	concatenation = tf.concat([displacement, displacement_feat, displacement_time], axis=3)
        	concatenation_2 = tf.concat([displacement_2, displacement_feat_2,displacement_time_2], axis=3)         

        #Unifty both concatenations
        concatenation = tf.concat([concatenation, concatenation_2], axis=2)
        #print("Concatenation.shape", concatenation.shape)

        
        # 5. Fully-connected layer (the only parameters)
        with tf.variable_scope('graph-rnn') as sc:
        	S1 = tf.layers.conv2d(inputs=concatenation, filters=out_channels, kernel_size=1, strides=1, padding='valid', data_format='channels_last', activation=activation, name='fc')
        
        #print("new S", S1.shape)
        #S1_before_Max = S1
        
        # 6. Pooling
        if pooling=='max':
        	S1 = tf.reduce_max(S1, axis=[2], keepdims=False)
        elif pooling=='avg':
        	S1 =tf.reduce_mean(S1, axis=[2], keepdims=False)  
        	
        return (P1, C1, F1, S1, T1, extra) 


""" ================================================== """
"""               GRAPH-FEATURE CELL                  """
""" ================================================== """

        
def graph_feat(P1,
              C1,
              F1,
              radius,
              nsample,
              out_channels,
              activation,
              knn=False,
              pooling='max',
              scope='graph_feat'):

    """
    Input:
        P1:     (batch_size, npoint, 3)
        C1:     (batch_size, npoint, feat_channels)
    Output:
        F1:     (batch_size, npoint, out_channels)
        S1:     (batch_size, npoint, out_channels) = None
    """
   
    # 1. Sample points
    if knn:
    	_, idx = knn_point(nsample, P1, P1)
    else:
        idx, cnt = query_ball_point(radius, nsample, P1, P1)
        _, idx_knn = knn_point(nsample, P1, P1)
        cnt = tf.tile(tf.expand_dims(cnt, -1), [1, 1, nsample])
        idx = tf.where(cnt > (nsample-1), idx, idx_knn)

    # 2.1 Group P2 points
    P1_grouped = group_point(P1, idx)                       # batch_size, npoint, nsample, 3
    #C1_grouped = group_point(C1, idx)
    
    # 3. Calcaulate displacements
    P1_expanded = tf.expand_dims(P1, 2)                     # batch_size, npoint, 1,       3
    displacement = P1_grouped - P1_expanded                 # batch_size, npoint, nsample, 3
    #C1_expanded = tf.expand_dims(C1, 2)                     # batch_size, npoint, 1,       3
    #displacement_color = C1_grouped - C1_expanded                 # batch_size, npoint, nsample, 3   


    # 4. Concatenate X1, S2 and displacement
    if F1 is not None:
    	F1_grouped = group_point(F1, idx)     
    	concatenation =  tf.concat([P1_grouped, F1_grouped], axis=3) 
    	concatenation = tf.concat([concatenation,displacement], axis=3)
    else:
    	concatenation = tf.concat([P1_grouped, displacement], axis=3)
    
    
    # 5. Fully-connected layer (the only parameters)
    with tf.variable_scope(scope) as sc:
        F1 = tf.layers.conv2d(inputs=concatenation, filters=out_channels, kernel_size=1, strides=1, padding='valid', data_format='channels_last', activation=activation, name='fc')

    # 6. Pooling

    if pooling=='max':
        F1= tf.reduce_max(F1, axis=[2], keepdims=False)
    elif pooling=='avg':
        F1= tf.reduce_mean(F1, axis=[2], keepdims=False)    

    return (F1)
            
class GraphFeatureCell(object):
    def __init__(self,
                 radius,
                 nsample,
                 out_channels,
                 knn=False,
                 pooling='max',
                 activation = None):

        self.radius = radius
        self.nsample = nsample
        self.out_channels = out_channels
        self.knn = knn
        self.pooling = pooling
        self.activation = activation


    def __call__(self, inputs):

        P1, C1, F1, S1 = inputs
        
        F1 = graph_feat(P1, C1, F1, radius=self.radius, nsample=self.nsample, out_channels=self.out_channels, knn=self.knn, pooling=self.pooling, activation = self.activation)
        
        
        return (P1, C1, F1, S1)

""" ================================================== """
"""               POINT-RNN ORIGINAL                   """
""" ================================================== """


def point_rnn_original(P1,
              P2,
              X1,
              S2,
              radius,
              nsample,
              out_channels,
              knn=False,
              pooling='max',
              scope='point_rnn'):
    """
    Input:
        P1:     (batch_size, npoint, 3)
        P2:     (batch_size, npoint, 3)
        X1:     (batch_size, npoint, feat_channels)
        S2:     (batch_size, npoint, out_channels)
    Output:
        S1:     (batch_size, npoint, out_channels)
    """
    # 1. Sample points
    if knn:
        _, idx = knn_point(nsample, P2, P1)
    else:
        idx, cnt = query_ball_point(radius, nsample, P2, P1)
        _, idx_knn = knn_point(nsample, P2, P1)
        cnt = tf.tile(tf.expand_dims(cnt, -1), [1, 1, nsample])
        idx = tf.where(cnt > (nsample-1), idx, idx_knn)

    # 2.1 Group P2 points
    P2_grouped = group_point(P2, idx)                       # batch_size, npoint, nsample, 3
    # 2.2 Group P2 states
    S2_grouped = group_point(S2, idx)                       # batch_size, npoint, nsample, out_channels

    # 3. Calcaulate displacements
    P1_expanded = tf.expand_dims(P1, 2)                     # batch_size, npoint, 1,       3
    displacement = P2_grouped - P1_expanded                 # batch_size, npoint, nsample, 3

    # 4. Concatenate X1, S2 and displacement
    if X1 is not None:
        X1_expanded = tf.tile(tf.expand_dims(X1, 2), [1, 1, nsample, 1])                # batch_size, npoint, sample,  feat_channels
        correlation = tf.concat([S2_grouped, X1_expanded], axis=3)                      # batch_size, npoint, nsample, feat_channels+out_channels
        correlation = tf.concat([correlation, displacement], axis=3)                    # batch_size, npoint, nsample, feat_channels+out_channels+3
    else:
        correlation = tf.concat([S2_grouped, displacement], axis=3)                     # batch_size, npoint, nsample, out_channels+3

    # 5. Fully-connected layer (the only parameters)
    with tf.variable_scope(scope) as sc:
        S1 = tf.layers.conv2d(inputs=correlation, filters=out_channels, kernel_size=1, strides=1, padding='valid', data_format='channels_last', activation=None, name='fc')

    # 6. Pooling
    return tf.reduce_max(S1, axis=[2], keepdims=False)


class PointRNNCell_original(object):
    def __init__(self,
                 radius,
                 nsample,
                 out_channels,
                 knn=False,
                 pooling='max'):

        self.radius = radius
        self.nsample = nsample
        self.out_channels = out_channels
        self.knn = knn
        self.pooling = pooling

    def init_state(self, inputs, state_initializer=tf.zeros_initializer(), dtype=tf.float32):
        """Helper function to create an initial state given inputs.
        Args:
            inputs: tube of (P, X). the first dimension P or X being batch_size
            state_initializer: Initializer(shape, dtype) for state Tensor.
            dtype: Optional dtype, needed when inputs is None.
        Returns:
            A tube of tensors representing the initial states.
        """
        # Handle both the dynamic shape as well as the inferred shape.
        P, X = inputs

        # inferred_batch_size = tf.shape(P)[0]
        inferred_batch_size = P.get_shape().with_rank_at_least(1)[0]
        inferred_npoints = P.get_shape().with_rank_at_least(1)[1]
        inferred_xyz_dimensions = P.get_shape().with_rank_at_least(1)[2]

        P = state_initializer([inferred_batch_size, inferred_npoints, inferred_xyz_dimensions], dtype=P.dtype)
        S = state_initializer([inferred_batch_size, inferred_npoints, self.out_channels], dtype=dtype)

        #print("P.shaoe", P.shape)
        #print("S.shape:", S.shape)
        return (P, S)

    def __call__(self, inputs, states):
        if states is None:
            states = self.init_state(inputs)

        P1, X1 = inputs
        P2, S2 = states

        S1 = point_rnn_original(P1, P2, X1, S2, radius=self.radius, nsample=self.nsample, out_channels=self.out_channels, knn=self.knn, pooling=self.pooling)

        #print("return P1", P1)
        #print("return S1", S1)

        return (P1, S1)


""" ================================================== """
"""               OG GRAPH-RNN without spatio-temporal  """
""" ================================================== """

class GraphRNN_WithoutSpatio_Cell(object):
    def __init__(self,
                 radius,
                 nsample,
                 out_channels,
                 knn=False,
                 pooling='max',
                 activation= None):

        self.radius = radius
        self.nsample = nsample
        self.out_channels = out_channels
        self.knn = knn
        self.pooling = pooling
        self.activation = activation

    def init_state(self, inputs, state_initializer=tf.zeros_initializer(), dtype=tf.float32):
        """Helper function to create an initial state given inputs.
        Args:
            inputs: tube of (P, X). the first dimension P or X being batch_size
            state_initializer: Initializer(shape, dtype) for state Tensor.
            dtype: Optional dtype, needed when inputs is None.
        Returns:
            A tube of tensors representing the initial states.
        """
        # Handle both the dynamic shape as well as the inferred shape.
        P, C, F, X, T = inputs

        # inferred_batch_size = tf.shape(P)[0]
        inferred_batch_size = P.get_shape().with_rank_at_least(1)[0]
        inferred_npoints = P.get_shape().with_rank_at_least(1)[1]
        inferred_xyz_dimensions = P.get_shape().with_rank_at_least(1)[2]
        inferred_feature_dimensions = 128 # ASSUMPTION
        
        P = state_initializer([inferred_batch_size, inferred_npoints, inferred_xyz_dimensions], dtype=P.dtype)
        #C = state_initializer([inferred_batch_size, inferred_npoints, inferred_xyz_dimensions], dtype=dtype)
        C = None
        #F = state_initializer([inferred_batch_size, inferred_npoints, inferred_xyz_dimensions], dtype=dtype)
        S = state_initializer([inferred_batch_size, inferred_npoints, self.out_channels], dtype=dtype)
        extra = None        

        return (P, C, F, S, T, extra)

    def __call__(self, inputs, states):
        if states is None:
            states = self.init_state(inputs)

        P1, C1, F1, X1, T1= inputs
        P2, C2, F2, S2, T2, extra = states
        
        radius = self.radius
        nsample = self.nsample
        out_channels = self.out_channels
        knn = self.knn
        pooling = self.pooling
        activation = self.activation
        
        print(" \nGraphRNN Without Spatio Operation")
        print("P1:",P1)  # time t
        print("P2:",P2)  # time t-1
        print("C1:",C1)
        print("C2:",C2)
        print("F1:",F1)
        print("F2:",F2)
        print("X1:",X1)
        print("S2:",S2)
        print("T1:",T1)
        print("T2:",T2)
        print("activation", activation)
        
                
        """
        print("Create adjacent matrix on feature space F1")
        P1_adj_matrix = tf_util.pairwise_distance(F1)
        print("P1_adj_matrix",P1_adj_matrix)
        P1_nn_idx = tf_util.knn(P1_adj_matrix, k= nsample)
        print("P1_nn_idx",P1_nn_idx)
        """
        
        # look at neighborhoodbood in P2
        print("Create adjacent matrix on feature space F2")
       	P2_adj_matrix = tf_util.pairwise_distance_2point_cloud(F2, F1)
       	print("P2_adj_matrix",P2_adj_matrix)
       	P2_nn_idx = tf_util.knn(P2_adj_matrix, k= nsample)
        print("P2_nn_idx",P2_nn_idx)
        
        if (knn == False) : # DO A BALL QUERY
        	print("\nBALL QUERY NOT IMPLEMENTED")
        	"""
        	idx, cnt = query_ball_point(radius, nsample, P1, P1)
        	cnt = tf.tile(tf.expand_dims(cnt, -1), [1, 1, nsample])
        	P1_nn_idx = tf.where(cnt > (nsample-1), idx, P1_nn_idx)
        	
        	idx, cnt = query_ball_point(radius, nsample, P2, P1)
        	cnt = tf.tile(tf.expand_dims(cnt, -1), [1, 1, nsample])
        	P2_nn_idx = tf.where(cnt > (nsample-1), idx, P2_nn_idx )
        	"""
        else:
        	print("KNN QUERY")
        
        P1_expanded = tf.expand_dims(P1, 2)
        F1_expanded = tf.expand_dims(F1, 2)
        T1_expanded = tf.expand_dims(T1, 2)
        
        # 2.1 Group P2 points
        P2_grouped = group_point(P2, P2_nn_idx)                      
        # 2.3 Group P color
        #C2_grouped = group_point(C2, P2_nn_idx)                       # batch_size, npoint, nsample, out_channels# 2.4 Group P feat
        F2_grouped = group_point(F2, P2_nn_idx)                       # batch_size, npoint, nsample, out_channels
        #2.4 Group S2 states
        S2_grouped = group_point(S2, P2_nn_idx)   
        # 2.4 Group P2 time
        T2_grouped = group_point(T2, P2_nn_idx)                       # batch_size, npoint, nsample, out_channels
 
              
        ##  Neighborhood P2 "
        # 3. Calculate displacements
        P2_expanded = tf.expand_dims(P2, 2)                     # batch_size, npoint, 1,       3
        displacement_2 = P2_grouped - P1_expanded                 # batch_size, npoint, nsample, 3
        #3.1 Calculate color displacements
        #C2_expanded = tf.expand_dims(C2, 2)                     # batch_size, npoint, 1,       3
        #displacement_color_2 = C2_grouped - C1_expanded           # batch_size, npoint, nsample, 3
        #3.1 Calculate feature displacements
        F2_expanded = tf.expand_dims(F2, 2)                     # batch_size, npoint, 1,       3
        displacement_feat_2 = F2_grouped - F1_expanded           # batch_size, npoint, nsample, 3        
        #3.1 Calculate time displacements
        T2_expanded = tf.expand_dims(T2, 2)                     # batch_size, npoint, 1,       3
        displacement_time_2 = T2_grouped - T1_expanded           # batch_size, npoint, nsample, 3

        # 4. Concatenate X1, S2 and displacement
        if X1 is not None:
        	print("Concatenation (t) = [F_i | S_ij | displacement_Pij | displacement_Fij| displacement_T] " )
        	X1_expanded = tf.tile(tf.expand_dims(X1, 2), [1, 1, nsample, 1])  
        	F1_expanded = tf.tile(tf.expand_dims(F1, 2), [1, 1, nsample, 1])                
        	
        	#concatenation = tf.concat([X1_expanded, S1_grouped], axis=3)         
        	#concatenation = tf.concat([concatenation, displacement ,displacement_feat, displacement_time], axis=3)
        	concatenation_2 = tf.concat([X1_expanded, S2_grouped], axis=3)
        	concatenation_2 = tf.concat([concatenation_2, displacement_2,displacement_feat_2, displacement_time_2], axis=3) 
        	             
        else:
        	print("Concatenation (t) = [displacement_Pij | displacement_Fij| displacement_T] " )

        	F1_expanded = tf.tile(tf.expand_dims(F1, 2), [1, 1, nsample, 1])                        
        	#concatenation = tf.concat([displacement, displacement_feat, displacement_time], axis=3)
        	concatenation_2 = tf.concat([displacement_2, displacement_feat_2,displacement_time_2], axis=3)         

        #Unifty both concatenations
        #concatenation = tf.concat([concatenation, concatenation_2], axis=2)
        concatenation = concatenation_2
        
        # 5. Fully-connected layer (the only parameters)
        with tf.variable_scope('graph-rnn') as sc:
        	S1 = tf.layers.conv2d(inputs=concatenation, filters=out_channels, kernel_size=1, strides=1, padding='valid', data_format='channels_last', activation=activation, name='fc')
        
        #S1_before_Max = S1
        
        # 6. Pooling
        if pooling=='max':
        	S1 = tf.reduce_max(S1, axis=[2], keepdims=False)
        elif pooling=='avg':
        	S1 =tf.reduce_mean(S1, axis=[2], keepdims=False)  
        	
        return (P1, C1, F1, S1, T1, extra) 


        
""" ================================================== 
               GRAPH-RNN OG   L2 on Kernel and L2 bias None activty           
 ================================================== """

class GraphRNNCell_L2(object):
    def __init__(self,
                 radius,
                 nsample,
                 out_channels,
                 knn=False,
                 pooling='max',
                 activation= None):

        self.radius = radius
        self.nsample = nsample
        self.out_channels = out_channels
        self.knn = knn
        self.pooling = pooling
        self.activation = activation

    def init_state(self, inputs, state_initializer=tf.zeros_initializer(), dtype=tf.float32):
        """Helper function to create an initial state given inputs.
        Args:
            inputs: tube of (P, X). the first dimension P or X being batch_size
            state_initializer: Initializer(shape, dtype) for state Tensor.
            dtype: Optional dtype, needed when inputs is None.
        Returns:
            A tube of tensors representing the initial states.
        """
        # Handle both the dynamic shape as well as the inferred shape.
        P, C, F, X, T = inputs

        # inferred_batch_size = tf.shape(P)[0]
        inferred_batch_size = P.get_shape().with_rank_at_least(1)[0]
        inferred_npoints = P.get_shape().with_rank_at_least(1)[1]
        inferred_xyz_dimensions = P.get_shape().with_rank_at_least(1)[2]
        #inferred_feature_dimensions = 128 # ASSUMPTION
        
        P = state_initializer([inferred_batch_size, inferred_npoints, inferred_xyz_dimensions], dtype=P.dtype)
        #C = state_initializer([inferred_batch_size, inferred_npoints, inferred_xyz_dimensions], dtype=dtype)
        C = None
        #F = state_initializer([inferred_batch_size, inferred_npoints, inferred_xyz_dimensions], dtype=dtype)
        S = state_initializer([inferred_batch_size, inferred_npoints, self.out_channels], dtype=dtype)
        extra = None        

        return (P, C, F, S, T, extra)

    def __call__(self, inputs, states):
        if states is None:
            states = self.init_state(inputs)

        P1, C1, F1, X1, T1= inputs
        P2, C2, F2, S2, T2, extra = states
        
        radius = self.radius
        nsample = self.nsample
        out_channels = self.out_channels
        knn = self.knn
        pooling = self.pooling
        activation = self.activation
        
        print(" \nGraphRNN Operation with regulaizers")
        print("P1:",P1)  # time t
        print("P2:",P2)  # time t-1
        print("C1:",C1)
        print("C2:",C2)
        print("F1:",F1)
        print("F2:",F2)
        print("X1:",X1)
        print("S2:",S2)
        print("T1:",T1)
        print("T2:",T2)
        print("activation", activation)
        
                
        print("Create adjacent matrix on feature space F1")
        P1_adj_matrix = tf_util.pairwise_distance(F1)
        print("P1_adj_matrix",P1_adj_matrix)
        P1_nn_idx = tf_util.knn(P1_adj_matrix, k= nsample)
        print("P1_nn_idx",P1_nn_idx)
        
        # look at neighborhoodbood in P2
        print("Create adjacent matrix on feature space F2")
       	P2_adj_matrix = tf_util.pairwise_distance_2point_cloud(F2, F1)
       	print("P2_adj_matrix",P2_adj_matrix)
       	P2_nn_idx = tf_util.knn(P2_adj_matrix, k= nsample)
        print("P2_nn_idx",P2_nn_idx)
        
        if (knn == False) : # DO A BALL QUERY
        	print("\nBALL QUERY NOT IMPLEMENTED")
        	"""
        	idx, cnt = query_ball_point(radius, nsample, P1, P1)
        	cnt = tf.tile(tf.expand_dims(cnt, -1), [1, 1, nsample])
        	P1_nn_idx = tf.where(cnt > (nsample-1), idx, P1_nn_idx)
        	
        	idx, cnt = query_ball_point(radius, nsample, P2, P1)
        	cnt = tf.tile(tf.expand_dims(cnt, -1), [1, 1, nsample])
        	P2_nn_idx = tf.where(cnt > (nsample-1), idx, P2_nn_idx )
        	"""
        else:
        	print("KNN QUERY")

        
        # 2.1 Group P1 points
        P1_grouped = group_point(P1, P1_nn_idx)                      
        # 2.3 Group P color
        #if (C1 is not None):
        	#C1_grouped = group_point(C1, P1_nn_idx)                       # batch_size, npoint, nsample, out_channels
        # 2.4 Group P feat
        F1_grouped = group_point(F1, P1_nn_idx)                       # batch_size, npoint, nsample, out_channels
        # 2.4 Group P time
        T1_grouped = group_point(T1, P1_nn_idx)                       # batch_size, npoint, nsample, out_channels
        # 2.2 Group P1 states
        if (X1 is not None):
        	S1_grouped = group_point(X1, P1_nn_idx)  
        
        # 2.1 Group P2 points
        P2_grouped = group_point(P2, P2_nn_idx)                      
        # 2.3 Group P color
        #C2_grouped = group_point(C2, P2_nn_idx)                       # batch_size, npoint, nsample, out_channels# 2.4 Group P feat
        F2_grouped = group_point(F2, P2_nn_idx)                       # batch_size, npoint, nsample, out_channels
        #2.4 Group S2 states
        S2_grouped = group_point(S2, P2_nn_idx)   
        # 2.4 Group P2 time
        T2_grouped = group_point(T2, P2_nn_idx)                       # batch_size, npoint, nsample, out_channels
 
        ##  Neighborhood P1"
        # 3. Calculate displacements
        P1_expanded = tf.expand_dims(P1, 2)                     # batch_size, npoint, 1,       3
        displacement = P1_grouped - P1_expanded                 # batch_size, npoint, nsample, 3
        #3.1 Calculate color displacements
        #C1_expanded = tf.expand_dims(C1, 2)                     # batch_size, npoint, 1,       3
        #displacement_color = C1_grouped - C1_expanded           # batch_size, npoint, nsample, 3
        #3.1 Calculate feature displacements
        F1_expanded = tf.expand_dims(F1, 2)                     # batch_size, npoint, 1,       3
        displacement_feat = F1_grouped - F1_expanded           # batch_size, npoint, nsample, 3
        #3.1 Calculate time displacements
        T1_expanded = tf.expand_dims(T1, 2)                     # batch_size, npoint, 1,       3
        displacement_time = T1_grouped - T1_expanded           # batch_size, npoint, nsample, 3
              
        ##  Neighborhood P2 "
        # 3. Calculate displacements
        P2_expanded = tf.expand_dims(P2, 2)                     # batch_size, npoint, 1,       3
        displacement_2 = P2_grouped - P1_expanded                 # batch_size, npoint, nsample, 3
        #3.1 Calculate color displacements
        #C2_expanded = tf.expand_dims(C2, 2)                     # batch_size, npoint, 1,       3
        #displacement_color_2 = C2_grouped - C1_expanded           # batch_size, npoint, nsample, 3
        #3.1 Calculate feature displacements
        F2_expanded = tf.expand_dims(F2, 2)                     # batch_size, npoint, 1,       3
        displacement_feat_2 = F2_grouped - F1_expanded           # batch_size, npoint, nsample, 3        
        #3.1 Calculate time displacements
        T2_expanded = tf.expand_dims(T2, 2)                     # batch_size, npoint, 1,       3
        displacement_time_2 = T2_grouped - T1_expanded           # batch_size, npoint, nsample, 3

        # 4. Concatenate X1, S2 and displacement
        if X1 is not None:
        	print("Concatenation (t) = [F_i | S_ij | displacement_Pij | displacement_Fij| displacement_T] " )
        	X1_expanded = tf.tile(tf.expand_dims(X1, 2), [1, 1, nsample, 1])  
        	F1_expanded = tf.tile(tf.expand_dims(F1, 2), [1, 1, nsample, 1])                
        	
        	concatenation = tf.concat([X1_expanded, S1_grouped], axis=3)         
        	concatenation = tf.concat([concatenation, displacement ,displacement_feat, displacement_time], axis=3)
        	concatenation_2 = tf.concat([X1_expanded, S2_grouped], axis=3)
        	concatenation_2 = tf.concat([concatenation_2, displacement_2,displacement_feat_2, displacement_time_2], axis=3) 
        	             
        else:
        	print("Concatenation (t) = [displacement_Pij | displacement_Fij| displacement_T] " )

        	F1_expanded = tf.tile(tf.expand_dims(F1, 2), [1, 1, nsample, 1])                        
        	concatenation = tf.concat([displacement, displacement_feat, displacement_time], axis=3)
        	concatenation_2 = tf.concat([displacement_2, displacement_feat_2,displacement_time_2], axis=3)         

        #Unifty both concatenations
        concatenation = tf.concat([concatenation, concatenation_2], axis=2)

        
        # 5. Fully-connected layer (the only parameters)
        with tf.variable_scope('graph-rnn') as sc:
        	S1 = tf.layers.conv2d(inputs=concatenation, filters=out_channels, kernel_size=1, strides=1, padding='valid', data_format='channels_last',bias_regularizer='l2',  kernel_regularizer= 'l2', activity_regularizer=None, activation=activation, name='fc')
        
        #S1_before_Max = S1
        
        # 6. Pooling
        if pooling=='max':
        	S1 = tf.reduce_max(S1, axis=[2], keepdims=False)
        elif pooling=='avg':
        	S1 =tf.reduce_mean(S1, axis=[2], keepdims=False)  
        	
        return (P1, C1, F1, S1, T1, extra) 


""" ================================================== """
"""               GRAPH-FEATURE CELL L2                  """
""" ================================================== """

        
def graph_feat_l2(P1,
              C1,
              F1,
              radius,
              nsample,
              out_channels,
              activation,
              knn=False,
              pooling='max',
              scope='graph_feat'):

    """
    Input:
        P1:     (batch_size, npoint, 3)
        C1:     (batch_size, npoint, feat_channels)
    Output:
        F1:     (batch_size, npoint, out_channels)
        S1:     (batch_size, npoint, out_channels) = None
    """
    print("Graph-Feat operation")
   
    # 1. Sample points
    if knn:
    	print("KNN query")
    	_, idx = knn_point(nsample, P1, P1)
    else:
        idx, cnt = query_ball_point(radius, nsample, P1, P1)
        _, idx_knn = knn_point(nsample, P1, P1)
        cnt = tf.tile(tf.expand_dims(cnt, -1), [1, 1, nsample])
        idx = tf.where(cnt > (nsample-1), idx, idx_knn)

    # 2.1 Group P2 points
    P1_grouped = group_point(P1, idx)                       # batch_size, npoint, nsample, 3
    #C1_grouped = group_point(C1, idx)
    
    # 3. Calcaulate displacements
    P1_expanded = tf.expand_dims(P1, 2)                     # batch_size, npoint, 1,       3
    displacement = P1_grouped - P1_expanded                 # batch_size, npoint, nsample, 3
    #C1_expanded = tf.expand_dims(C1, 2)                     # batch_size, npoint, 1,       3
    #displacement_color = C1_grouped - C1_expanded                 # batch_size, npoint, nsample, 3   


    # 4. Concatenate X1, S2 and displacement
    if F1 is not None:
    	F1_grouped = group_point(F1, idx)     
    	concatenation =  tf.concat([P1_grouped, F1_grouped], axis=3) 
    	concatenation = tf.concat([concatenation,displacement], axis=3)
    	print("Concatenation = [P_i | F_i | displacement_Pij] " ) 
    else:
    	concatenation = tf.concat([P1_grouped, displacement], axis=3)
    	print("Concatenation = [P_i | displacement_Pij] " ) 
    
    #print("Concatenation",concatenation)
    
    # 5. Fully-connected layer (the only parameters)
    with tf.variable_scope(scope) as sc:
        F1 = tf.layers.conv2d(inputs=concatenation, filters=out_channels, kernel_size=1, strides=1, padding='valid', data_format='channels_last', activation=activation, bias_regularizer='l2',  kernel_regularizer= 'l2', activity_regularizer=None , name='fc')

    # 6. Pooling

    if pooling=='max':
        F1= tf.reduce_max(F1, axis=[2], keepdims=False)
    elif pooling=='avg':
        F1= tf.reduce_mean(F1, axis=[2], keepdims=False)    

    return (F1)
            
""" Interpolate only the features """
def copy_feat(xyz1, xyz2, feat2):
	 
	 
	print("\n Upsmaple features Module" )
	"""
	Input:
	  xyz1: 4000 points
	  xyz2: 1000 points
	  feat: 1000 feat
	Output:
	 interpolated feat: 4000 feat
	"""
	
	k= 4
	dist, idx = knn_point(k, xyz2 ,xyz1)
	feat_grouped = group_point(feat2, idx)
	xyz2_grouped = group_point(xyz2, idx)

	
	#xyz1_expanded = tf.expand_dims(xyz1, 2)  
	#displament
	#displacement = xyz2_grouped - xyz1_expanded

	#dist = tf.norm(displacement, ord='euclidean', axis=3, keepdims=True, name=None)
	#dist= tf.nn.softmax(dist)
	#print("dist", dist)
	#print("feat_grouped", feat_grouped)
	
	#weighted_feat_grouped = feat_grouped *dist
	#print("weighted_feat_grouped", weighted_feat_grouped)


	#feat_grouped= tf.reduce_mean(feat_grouped, axis=[2], keepdims=False)  
	interpolated_feat = feat_grouped #* (1/dist)
	#interpolated_feat = weighted_feat_grouped #* (1/dist)
	interpolated_feat= tf.reduce_mean(interpolated_feat, axis=[2], keepdims=False) 
	
	return interpolated_feat   
	         	     
def copy_feat_2(xyz1, xyz2, feat2):
	 
	 
	print("\n Upsmaple features Module" )
	"""
	Input:
	  xyz1: 4000 points
	  xyz2: 1000 points
	  feat2: 1000 feat
	Output:
	 interpolated feat: 4000 feat
	"""
	
	k= 10
	dist, idx = knn_point(k, xyz2 ,xyz1)
	feat_grouped = group_point(feat2, idx)
	xyz2_grouped = group_point(xyz2, idx)

	
	xyz1_expanded = tf.expand_dims(xyz1, 2)  
	#displament
	displacement = xyz2_grouped - xyz1_expanded

	dist = tf.norm(displacement, ord='euclidean', axis=3, keepdims=True, name=None)
	dist= tf.nn.softmax(dist)
	print("dist", dist)
	print("feat_grouped", feat_grouped)
	
	weighted_feat_grouped = feat_grouped *dist
	print("weighted_feat_grouped", weighted_feat_grouped)


	#feat_grouped= tf.reduce_mean(feat_grouped, axis=[2], keepdims=False)  
	#interpolated_feat = feat_grouped #* (1/dist)
	interpolated_feat = weighted_feat_grouped #* (1/dist)
	interpolated_feat= tf.reduce_mean(interpolated_feat, axis=[2], keepdims=False) 
	
	return interpolated_feat   
	         	     

""" ================================================== 
              GRAPH-RNN ATTENTION STATES COMBINATION                      
    ================================================== """

        
def GraphAttention_States_Combination(P0,
              P1,
              P2,
              P3,
              S1,
              S2,
              S3,
              nsample,
              activation,
              out_channels,
              scope='attention_states_combination'):

    """
    Learn Sf for all P0 points
    Input:
        S1:     (batch_size, npoint, 3)
        S2:     (batch_size, npoint, feat_channels)
        S3:
    Output:
        Sf:     (batch_size, npoint, out_channels)
    """
    print(" ==  Graph Attention States Combination ===")
    
    print("S1", S1)
    print("S2", S2)
    print("S3", S3)
    print("activation", activation)
    
    # 2. Learn Attention Weights
    
    #For states 1
    concatenation = tf.concat( [S3,S1], axis = 2)
    print("concatenation", concatenation)
    with tf.variable_scope('attention_s1', reuse=tf.AUTO_REUSE) as scope:
    	att_s1 = tf.layers.conv1d(inputs=concatenation, filters=1, kernel_size=1, strides=1, padding='valid', data_format='channels_last', activation=activation, name='at_s1')	    	    
    #For states 2
    concatenation = tf.concat( [S3,S2], axis = 2)
    print("concatenation", concatenation)
    with tf.variable_scope('attention_s2', reuse=tf.AUTO_REUSE) as scope:
    	att_s2 = tf.layers.conv1d(inputs=concatenation, filters=1, kernel_size=1, strides=1, padding='valid', data_format='channels_last', activation=activation, name='at_s2')

    print("att_s2", att_s2.shape)
    print("att_s1", att_s1.shape)
        
    
    # 3 .Process States S1 & S2
    with tf.variable_scope('psi_S1', reuse=tf.AUTO_REUSE) as scope:
    	psi_S1 = tf.layers.conv1d(inputs=S1, filters=out_channels, kernel_size=1, strides=1, padding='valid', data_format='channels_last', activation=activation, name='psi_s1')

    with tf.variable_scope('psi_S2', reuse=tf.AUTO_REUSE) as scope:
    	psi_S2 = tf.layers.conv1d(inputs=S2, filters=out_channels, kernel_size=1, strides=1, padding='valid', data_format='channels_last', activation=activation, name='psi_s2')
    	    	     
    	    	     
    # Multiply processed state by the attention
    print("psi_S1", psi_S1.shape)
    print("psi_S2", psi_S2.shape)
    
    S1 = att_s1 * psi_S1
    S2 = att_s2 * psi_S2    
    print("S1", S1)
    
    # 4. Aggregation 
    S1 = tf.expand_dims(S1, axis =2)
    S2 = tf.expand_dims(S2, axis =2)
    print("S1", S1)
    S_agrregation = tf.concat( [S1, S2], axis=2)
    print("S_agrregation", S_agrregation)    
    #Use mean
    S_agrregation = tf.reduce_mean(S_agrregation, axis =2)
    print("S_agrregation", S_agrregation)    
    
    
    # 5. Process S3 and S_agreggation together
    concatention = tf.concat( [S3, S_agrregation], axis=2)
    #print("concatention", concatention)
    with tf.variable_scope('Sf_layer', reuse=tf.AUTO_REUSE) as scope:
    	Sf = tf.layers.conv1d(inputs= concatention, filters=out_channels, kernel_size=1, strides=1, padding='valid', data_format='channels_last', activation=tf.nn.relu, name='final_Sf')    
    
    print("Sf", Sf)
    
    return Sf

""" ================================================== 
           EFFICIENT GRAPH-RNN ATTENTION STATES COMBINATION                      
    ================================================== """
       
def Efficient_GraphAttention_States_Combination(P0,
              P1,
              P2,
              P3,
              S1,
              S2,
              S3,
              nsample,
              activation,
              out_channels,
              scope='attention_states_combination'):

    """
    Learn Sf for all P0 points
    Input:
        S1:     (batch_size, npoint, 3)
        S2:     (batch_size, npoint, feat_channels)
        S3:     
    Output:
        Sf:     (batch_size, npoint, out_channels)
    """
    print(" ==  Graph Efficient Attention States Combination ===")
    
    print("P0", P0)
    print("P1", P1)
    print("P2", P2)
    print("P3", P3)

    print("S1", S1)
    print("S2", S2)
    print("S3", S3)
    print("")
    
    # 1. Mathch point between the point clouds.
    """
    Geo_Adj = tf_util.pairwise_distance_2point_cloud(P3, P1)
    Geo_Adj= Geo_Adj[0]
    P1_idx = tf_util.knn(Geo_Adj, k= nsample)
    print("P1_idx", P1_idx)
    # return for each point in P1 the neighboorhos in P3
    """
    
    # 1.3 Find point in P1
    dist, P1_idx = knn_point(nsample, P3 ,P1) # For each point in P0 find the k-cloest point in P3
    #print("dist", dist)
    print("P1_idx", P1_idx)
    
    S3_grouped_to_P1 = group_point(S3, P1_idx)
    print("S3_grouped_to_P1", S3_grouped_to_P1)    

    # 1.3 Find point in P2
    dist, P2_idx = knn_point(nsample, P3 ,P2) # For each point in P0 find the k-cloest point in P3
    S3_grouped_to_P2 = group_point(S3, P2_idx)
    print("S3_grouped_to_P2", S3_grouped_to_P2)  
    
    
    #1.1 Expand points S3 to match dimension
    S1_expanded = tf.tile(tf.expand_dims(S1, 2), [1, 1, nsample, 1])
    print("S1_expanded", S1_expanded)    
    S2_expanded = tf.tile(tf.expand_dims(S2, 2), [1, 1, nsample, 1])
    print("S2_expanded", S2_expanded)    	
    print("")
        
    
    # 2. Learn Attention Weights
    print(" -- Learn Attention --")
    #For states 1
    concatenation = tf.concat( [S3_grouped_to_P1,S1_expanded], axis = 3)
    print("concatenation", concatenation)
    with tf.variable_scope('attention_s1', reuse=tf.AUTO_REUSE) as scope:
    	att_s1 = tf.layers.conv2d(inputs=concatenation, filters=1, kernel_size=1, strides=1, padding='valid', data_format='channels_last', activation=activation, name='at_s1')
    print("att_s1", att_s1.shape)	 	    	    
    #For states 2
    concatenation = tf.concat( [S3_grouped_to_P2,S2_expanded], axis = 3)
    print("concatenation", concatenation)
    with tf.variable_scope('attention_s2', reuse=tf.AUTO_REUSE) as scope:
    	att_s2 = tf.layers.conv2d(inputs=concatenation, filters=1, kernel_size=1, strides=1, padding='valid', data_format='channels_last', activation=activation, name='at_s2')

    print("att_s2", att_s2.shape)
    print("att_s1", att_s1.shape)
        
    
    # 3 .Process States S1 & S2
    with tf.variable_scope('psi_S1', reuse=tf.AUTO_REUSE) as scope:
    	psi_S1 = tf.layers.conv1d(inputs=S1, filters=out_channels, kernel_size=1, strides=1, padding='valid', data_format='channels_last', activation=activation, name='psi_s1')

    with tf.variable_scope('psi_S2', reuse=tf.AUTO_REUSE) as scope:
    	psi_S2 = tf.layers.conv1d(inputs=S2, filters=out_channels, kernel_size=1, strides=1, padding='valid', data_format='channels_last', activation=activation, name='psi_s2')
    	    	     
    	    	     
    # Multiply processed state by the attention
    print("psi_S1", psi_S1.shape)
    print("psi_S2", psi_S2.shape)
    
    print(" -- only one neighborhood (reshape attention) --")
    att_s1 = tf.squeeze(att_s1, axis=2)
    att_s2 = tf.squeeze(att_s2, axis=2)

    print("att_s2", att_s2.shape)
    print("att_s1", att_s1.shape)
        
    
    #psi_S1 = tf.tile(tf.expand_dims(psi_S1, 2), [1, 1, nsample, 1])
    #psi_S2 = tf.tile(tf.expand_dims(psi_S2, 2), [1, 1, nsample, 1])
    
    S1 = att_s1 * psi_S1
    S2 = att_s2 * psi_S2    
    print("S1", S1)
    print("S2", S2)
    print("")
    
    # 4. Aggregation 
    # 4.1 Find point in P1 that match P0
    dist, P1_idx = knn_point(nsample, P1 ,P0) # For each point in P0 find the k-cloest point in P3
    print("P1_idx", P1_idx)
    S1_grouped_to_P0 = group_point(S1, P1_idx)
    print("S1_grouped_to_P0", S1_grouped_to_P0)  
    
    # 4.1 Find point in P2 that match P0
    dist, P2_idx = knn_point(nsample, P2 ,P0) # For each point in P0 find the k-cloest point in P3
    print("P2_idx", P2_idx)
    S2_grouped_to_P0 = group_point(S2, P2_idx)
    print("S2_grouped_to_P0", S2_grouped_to_P0)   

    # 4.1 Find point in P3 that match P0
    dist, P3_idx = knn_point(nsample, P3 ,P0) # For each point in P0 find the k-cloest point in P3
    print("P3_idx", P3_idx)
    S3_grouped_to_P0 = group_point(S3, P3_idx)
    print("S3_grouped_to_P0", S3_grouped_to_P0)
    S3_grouped_to_P0 = tf.squeeze(S3_grouped_to_P0, axis=2)   
    
        
    # Reduce mean    
    S_agrregation = tf.concat( [S1_grouped_to_P0, S2_grouped_to_P0], axis=2)
    print("S_agrregation", S_agrregation)    
    #Use mean
    S_agrregation = tf.reduce_mean(S_agrregation, axis =2)
    print("S_agrregation", S_agrregation)    
    
    
    # 5. Process S3 and S_agreggation together
    concatention = tf.concat( [S3_grouped_to_P0, S_agrregation], axis=2)
    #print("concatention", concatention)
    with tf.variable_scope('Sf_layer', reuse=tf.AUTO_REUSE) as scope:
    	Sf = tf.layers.conv1d(inputs= concatention, filters=out_channels, kernel_size=1, strides=1, padding='valid', data_format='channels_last', activation=tf.nn.relu, name='final_Sf')    
    
    print("Sf", Sf)
    
    return (Sf, att_s1, att_s2) 



def Efficient_GraphAttention_States_Combination_concatenation(P0,
              P1,
              P2,
              P3,
              S1,
              S2,
              S3,
              nsample,
              activation,
              out_channels,
              scope='attention_states_combination'):

    """
    Learn Sf for all P0 points
    Input:
        S1:     (batch_size, npoint, 3)
        S2:     (batch_size, npoint, feat_channels)
        S3:     
    Output:
        Sf:     (batch_size, npoint, out_channels)
    """
    print(" ==  Graph Efficient Attention States Combination ===")
    
    print("P0", P0)
    print("P1", P1)
    print("P2", P2)
    print("P3", P3)

    print("S1", S1)
    print("S2", S2)
    print("S3", S3)
    print("")
    
    # 1. Mathch point between the point clouds.
    """
    Geo_Adj = tf_util.pairwise_distance_2point_cloud(P3, P1)
    Geo_Adj= Geo_Adj[0]
    P1_idx = tf_util.knn(Geo_Adj, k= nsample)
    print("P1_idx", P1_idx)
    # return for each point in P1 the neighboorhos in P3
    """
    
    # 1.3 Find point in P1
    dist, P1_idx = knn_point(nsample, P3 ,P1) # For each point in P0 find the k-cloest point in P3
    #print("dist", dist)
    print("P1_idx", P1_idx)
    
    S3_grouped_to_P1 = group_point(S3, P1_idx)
    print("S3_grouped_to_P1", S3_grouped_to_P1)    

    # 1.3 Find point in P2
    dist, P2_idx = knn_point(nsample, P3 ,P2) # For each point in P0 find the k-cloest point in P3
    S3_grouped_to_P2 = group_point(S3, P2_idx)
    print("S3_grouped_to_P2", S3_grouped_to_P2)  
    
    
    #1.1 Expand points S3 to match dimension
    S1_expanded = tf.tile(tf.expand_dims(S1, 2), [1, 1, nsample, 1])
    print("S1_expanded", S1_expanded)    
    S2_expanded = tf.tile(tf.expand_dims(S2, 2), [1, 1, nsample, 1])
    print("S2_expanded", S2_expanded)    	
    print("")
        
    
    # 2. Learn Attention Weights
    print(" -- Learn Attention --")
    #For states 1
    concatenation = tf.concat( [S3_grouped_to_P1,S1_expanded], axis = 3)
    print("concatenation", concatenation)
    with tf.variable_scope('attention_s1', reuse=tf.AUTO_REUSE) as scope:
    	att_s1 = tf.layers.conv2d(inputs=concatenation, filters=1, kernel_size=1, strides=1, padding='valid', data_format='channels_last', activation=activation, name='at_s1')
    print("att_s1", att_s1.shape)	 	    	    
    #For states 2
    concatenation = tf.concat( [S3_grouped_to_P2,S2_expanded], axis = 3)
    print("concatenation", concatenation)
    with tf.variable_scope('attention_s2', reuse=tf.AUTO_REUSE) as scope:
    	att_s2 = tf.layers.conv2d(inputs=concatenation, filters=1, kernel_size=1, strides=1, padding='valid', data_format='channels_last', activation=activation, name='at_s2')

    print("att_s2", att_s2.shape)
    print("att_s1", att_s1.shape)
        
    
    # 3 .Process States S1 & S2
    with tf.variable_scope('psi_S1', reuse=tf.AUTO_REUSE) as scope:
    	psi_S1 = tf.layers.conv1d(inputs=S1, filters=out_channels, kernel_size=1, strides=1, padding='valid', data_format='channels_last', activation=activation, name='psi_s1')

    with tf.variable_scope('psi_S2', reuse=tf.AUTO_REUSE) as scope:
    	psi_S2 = tf.layers.conv1d(inputs=S2, filters=out_channels, kernel_size=1, strides=1, padding='valid', data_format='channels_last', activation=activation, name='psi_s2')
    	    	     
    	    	     
    # Multiply processed state by the attention
    print("psi_S1", psi_S1.shape)
    print("psi_S2", psi_S2.shape)
    
    print(" -- only one neighborhood (reshape attention) --")
    att_s1 = tf.squeeze(att_s1, axis=2)
    att_s2 = tf.squeeze(att_s2, axis=2)

    print("att_s2", att_s2.shape)
    print("att_s1", att_s1.shape)
        
    
    #psi_S1 = tf.tile(tf.expand_dims(psi_S1, 2), [1, 1, nsample, 1])
    #psi_S2 = tf.tile(tf.expand_dims(psi_S2, 2), [1, 1, nsample, 1])
    
    S1 = att_s1 * psi_S1
    S2 = att_s2 * psi_S2    
    print("S1", S1)
    print("S2", S2)
    print("")
    
    # 4. Aggregation 
    # 4.1 Find point in P1 that match P0
    dist, P1_idx = knn_point(nsample, P1 ,P0) # For each point in P0 find the k-cloest point in P3
    print("P1_idx", P1_idx)
    S1_grouped_to_P0 = group_point(S1, P1_idx)
    print("S1_grouped_to_P0", S1_grouped_to_P0)  
    
    # 4.1 Find point in P2 that match P0
    dist, P2_idx = knn_point(nsample, P2 ,P0) # For each point in P0 find the k-cloest point in P3
    print("P2_idx", P2_idx)
    S2_grouped_to_P0 = group_point(S2, P2_idx)
    print("S2_grouped_to_P0", S2_grouped_to_P0)   

    # 4.1 Find point in P3 that match P0
    dist, P3_idx = knn_point(nsample, P3 ,P0) # For each point in P0 find the k-cloest point in P3
    print("P3_idx", P3_idx)
    S3_grouped_to_P0 = group_point(S3, P3_idx)
    print("S3_grouped_to_P0", S3_grouped_to_P0)
    S3_grouped_to_P0 = tf.squeeze(S3_grouped_to_P0, axis=2)   
    
        
    # Reduce mean    
    #S_agrregation = tf.concat( [S1_grouped_to_P0, S2_grouped_to_P0], axis=2)
    #print("S_agrregation", S_agrregation)    
    #Use mean
    #S_agrregation = tf.reduce_mean(S_agrregation, axis =2)
    #print("S_agrregation", S_agrregation)    
    

    #Do huge concatenation
    S_agrregation = tf.concat( [S1_grouped_to_P0, S2_grouped_to_P0], axis=3)
    print("S_agrregation", S_agrregation)
    #S_agrregation = tf.squeeze(S_agrregation)
    S_agrregation =  tf.reshape(S_agrregation, (S_agrregation.shape[0], S_agrregation.shape[1], S_agrregation.shape[3]) )
    print("S_agrregation", S_agrregation)
    


    # 5. Process S3 and S_agreggation together
    concatention = tf.concat( [S3_grouped_to_P0, S_agrregation], axis=2)
    #print("concatention", concatention)
    with tf.variable_scope('Sf_layer', reuse=tf.AUTO_REUSE) as scope:
    	Sf = tf.layers.conv1d(inputs= concatention, filters=out_channels, kernel_size=1, strides=1, padding='valid', data_format='channels_last', activation=tf.nn.relu, name='final_Sf')    
    
    print("Sf", Sf)
    
    return (Sf, att_s1, att_s2) 



""" ================================================== 
           EFFICIENT GRAPH-RNN ATTENTION STATES COMBINATION                      
    ================================================== """

        
def More_Efficient_GraphAttention_States_Combination(P0,
              P1,
              P2,
              P3,
              S1,
              S2,
              S3,
              nsample,
              activation,
              out_channels,
              scope='attention_states_combination'):

    """
    Learn Sf for all P0 points
    Input:
        S1:     (batch_size, npoint, 3)
        S2:     (batch_size, npoint, feat_channels)
        S3:     
    Output:
        Sf:     (batch_size, npoint, out_channels)
    """
    print(" ==  MORE Graph Efficient Attention States Combination ===")
    
    print("P0", P0)
    print("P1", P1)
    print("P2", P2)
    print("P3", P3)

    print("S1", S1)
    print("S2", S2)
    print("S3", S3)
    print("")
    

    
    # 1.3 Find point in P1
    dist, P1_idx = knn_point(nsample, P3 ,P1) # For each point in P1 find the k-cloest point in P3
    #print("dist", dist)
    print("P1_idx", P1_idx)
    S3_grouped_to_P1 = group_point(S3, P1_idx)
    print("S3_grouped_to_P1", S3_grouped_to_P1)    

    # 1.3 Find point in P2
    dist, P2_idx = knn_point(nsample, P3 ,P2) # For each point in P0 find the k-cloest point in P3
    S3_grouped_to_P2 = group_point(S3, P2_idx)
    print("S3_grouped_to_P2", S3_grouped_to_P2)  
    
    
    #1.1 Expand points S3 to match dimension
    S1_expanded = tf.tile(tf.expand_dims(S1, 2), [1, 1, nsample, 1])
    print("S1_expanded", S1_expanded)    
    S2_expanded = tf.tile(tf.expand_dims(S2, 2), [1, 1, nsample, 1])
    print("S2_expanded", S2_expanded)    	
    print("")
        
    
    # 2. Learn Attention Weights
    print(" -- Learn Attention --")
    #For states 1
    concatenation = tf.concat( [S3_grouped_to_P1,S1_expanded], axis = 3)
    print("concatenation", concatenation)
    with tf.variable_scope('attention_s1', reuse=tf.AUTO_REUSE) as scope:
    	att_s1 = tf.layers.conv2d(inputs=concatenation, filters=1, kernel_size=1, strides=1, padding='valid', data_format='channels_last', activation=activation, name='at_s1')
    print("att_s1", att_s1.shape)	 	    	    
    #For states 2
    concatenation = tf.concat( [S3_grouped_to_P2,S2_expanded], axis = 3)
    print("concatenation", concatenation)
    with tf.variable_scope('attention_s2', reuse=tf.AUTO_REUSE) as scope:
    	att_s2 = tf.layers.conv2d(inputs=concatenation, filters=1, kernel_size=1, strides=1, padding='valid', data_format='channels_last', activation=activation, name='at_s2')

    print("att_s2", att_s2.shape)
    print("att_s1", att_s1.shape)
        
    
    
    # 3 .Process States S1 & S2
    with tf.variable_scope('psi_S1', reuse=tf.AUTO_REUSE) as scope:
    	psi_S1 = tf.layers.conv1d(inputs=S1, filters=out_channels, kernel_size=1, strides=1, padding='valid', data_format='channels_last', activation=activation, name='psi_s1')

    with tf.variable_scope('psi_S2', reuse=tf.AUTO_REUSE) as scope:
    	psi_S2 = tf.layers.conv1d(inputs=S2, filters=out_channels, kernel_size=1, strides=1, padding='valid', data_format='channels_last', activation=activation, name='psi_s2')
    	    	     
    	    	     
    	    	     
    # Multiply processed state by the attention
    print("psi_S1", psi_S1.shape)
    print("psi_S2", psi_S2.shape)
    
    print(" -- only one neighborhood (reshape attention) --")
    att_s1 = tf.squeeze(att_s1, axis=2)
    att_s2 = tf.squeeze(att_s2, axis=2)

    print("att_s2", att_s2.shape)
    print("att_s1", att_s1.shape)
        
    
    #psi_S1 = tf.tile(tf.expand_dims(psi_S1, 2), [1, 1, nsample, 1])
    #psi_S2 = tf.tile(tf.expand_dims(psi_S2, 2), [1, 1, nsample, 1])
    
    S1 = att_s1 * psi_S1
    S2 = att_s2 * psi_S2    
    print("S1", S1)
    print("S2", S2)
    print("")

    #SET STUFF TO ZERO
    #S1 = S1 *0
    #S2 = S2 *0
    #S3 = S3 *0

    
    # 4. Aggregation 
    print(" --- Aggregation --")
    # 4.1 Find point in P1 that match P3 
    #dist, P1_idx = knn_point(nsample, P1 ,P0) # For each point in P0 find the k-cloest point in P3
    #print("P1_idx", P1_idx)
    #S3_grouped_to_P1 = group_point(S1, P1_idx)
    print("S3_grouped_to_P1", S3_grouped_to_P1)  

    # 4.1 Find point in P1 that match P2
    dist, P21_idx = knn_point(nsample, P2 ,P1) # For each point in P0 find the k-cloest point in P3
    S2_grouped_to_P1 = group_point(S2, P21_idx)
    print("S2_grouped_to_P1", S2_grouped_to_P1)  
    
    

    print("S1", S1)
    
    S1_expanded = tf.tile(tf.expand_dims(S1, 2), [1, 1, nsample, 1])
    
    print("S1", S1)
        
    print("\n -- do mean --")
    # Reduce mean    
    S_agrregation = tf.concat( [S1_expanded, S2_grouped_to_P1], axis=2)
    print("S_agrregation", S_agrregation)    
    #Use mean
    S_agrregation = tf.reduce_mean(S_agrregation, axis =2)
    print("S_agrregation", S_agrregation)    
    
    S3_grouped_to_P1 = tf.squeeze(S3_grouped_to_P1, axis=2)   
    
    
    # 5. Process S3 and S_agreggation together
    concatention = tf.concat( [S3_grouped_to_P1, S_agrregation], axis=2)
    print("concatenation", concatenation)
    #print("concatention", concatention)
    with tf.variable_scope('Sf_layer', reuse=tf.AUTO_REUSE) as scope:
    	S0 = tf.layers.conv1d(inputs= concatention, filters=out_channels, kernel_size=1, strides=1, padding='valid', data_format='channels_last', activation=tf.nn.relu, name='final_Sf')    
    
    print("S0", S0)
    
    
    # 6.Extrapolate Learned states to a higher resolution point cloud"
    print("\nExtrapolate Learned states to a higher resolution point cloud")
    dist, P0_idx = knn_point(nsample, P1 ,P0) # For each point in P0 find the k-cloest point in P3
    print("P0_idx", P0_idx)
    S0_grouped_to_P0 = group_point(S0, P0_idx)
    print("S0_grouped_to_P0", S0_grouped_to_P0)    
    
    
    Sf = tf.reduce_mean(S0_grouped_to_P0, axis =2 )
    print("Sf", Sf)


    
    return Sf
     


def Efficient_GraphAttention_States_Combination_Full_Attention(P0,
              P1,
              P2,
              P3,
              S1,
              S2,
              S3,
              nsample,
              activation,
              out_channels,
              scope='attention_states_combination'):

    """
    Learn Sf for all P0 points
    Input:
        S1:     (batch_size, npoint, 3)
        S2:     (batch_size, npoint, feat_channels)
        S3:     
    Output:
        Sf:     (batch_size, npoint, out_channels)
    """
    print(" ==  Graph Efficient Attention FULL ATTENTION Combination ===")
    
    print("P0", P0)
    print("P1", P1)
    print("P2", P2)
    print("P3", P3)

    print("S1", S1)
    print("S2", S2)
    print("S3", S3)
    print("")
    
    # 1. Mathch point between the point clouds.
    """
    Geo_Adj = tf_util.pairwise_distance_2point_cloud(P3, P1)
    Geo_Adj= Geo_Adj[0]
    P1_idx = tf_util.knn(Geo_Adj, k= nsample)
    print("P1_idx", P1_idx)
    # return for each point in P1 the neighboorhos in P3
    """
    
    
    # Grouping for P1
    # 1.3 Find point in P1 for S3
    dist, idx = knn_point(nsample, P3 ,P1) # For each point in P1 find the k-cloest point in S3
    print("idx - ", idx)
    S3_grouped_to_P1 = group_point(S3, idx)
    print("S3_grouped_to_P1", S3_grouped_to_P1)    


    dist, idx = knn_point(nsample, P2 ,P1) # For each point in P1 find the k-cloest point in S3
    S2_grouped_to_P1 = group_point(S2, idx)
    print("S2_grouped_to_P1", S2_grouped_to_P1)    
    
    
    # Grouping for S2
    # 1.3 Find point in P1 for S2
    dist, idx = knn_point(nsample, P3 ,P2) # For each point in P2 find the k-cloest point in S3
    S3_grouped_to_P2 = group_point(S3, idx)
    print("S3_grouped_to_P2", S3_grouped_to_P2)  

    dist, idx = knn_point(nsample, P1 ,P2) # For each point in P2 find the k-cloest point in S3
    S1_grouped_to_P2 = group_point(S1, idx)
    print("S1_grouped_to_P2", S1_grouped_to_P2)  
    
    
    #Grouping for S3

    _, idx = knn_point(nsample, P2 ,P3) # For each point in P2 find the k-cloest point in S3
    S2_grouped_to_P3 = group_point(S2, idx)
    print("S2_grouped_to_P3", S2_grouped_to_P3)  

    _, idx = knn_point(nsample, P1 ,P3) # For each point in P2 find the k-cloest point in S3
    S1_grouped_to_P3 = group_point(S1, idx)
    print("S1_grouped_to_P3", S1_grouped_to_P3)  
    
    
    #1.1 Expand points S3 to match dimension
    S1_expanded = tf.tile(tf.expand_dims(S1, 2), [1, 1, nsample, 1])
    print("S1_expanded", S1_expanded)    
    S2_expanded = tf.tile(tf.expand_dims(S2, 2), [1, 1, nsample, 1])
    print("S2_expanded", S2_expanded)    	
    S3_expanded = tf.tile(tf.expand_dims(S3, 2), [1, 1, nsample, 1])
    print("S3_expanded", S2_expanded)    
    print("")
        
    
    # 2. Learn Attention Weights
    print(" -- Learn Attention --")
    #For states 1
    concatenation = tf.concat( [S3_grouped_to_P1,S2_grouped_to_P1,S1_expanded], axis = 3)
    print("concatenation", concatenation)
    with tf.variable_scope('attention_s1', reuse=tf.AUTO_REUSE) as scope:
    	att_s1 = tf.layers.conv2d(inputs=concatenation, filters=1, kernel_size=1, strides=1, padding='valid', data_format='channels_last', activation=activation, name='at_s1')
    
    #For states 2
    concatenation = tf.concat( [S3_grouped_to_P2,S1_grouped_to_P2,S2_expanded], axis = 3)
    print("concatenation", concatenation)
    with tf.variable_scope('attention_s2', reuse=tf.AUTO_REUSE) as scope:
    	att_s2 = tf.layers.conv2d(inputs=concatenation, filters=1, kernel_size=1, strides=1, padding='valid', data_format='channels_last', activation=activation, name='at_s2')


    #For states 3
    concatenation = tf.concat( [S1_grouped_to_P3,S2_grouped_to_P3,S3_expanded], axis = 3)
    print("concatenation", concatenation)
    with tf.variable_scope('attention_s3', reuse=tf.AUTO_REUSE) as scope:
    	att_s3 = tf.layers.conv2d(inputs=concatenation, filters=1, kernel_size=1, strides=1, padding='valid', data_format='channels_last', activation=activation, name='at_s2') 

    print("att_s1", att_s1.shape)
    print("att_s2", att_s2.shape)
    print("att_s3", att_s3.shape)
            
    
    # 3 .Process States S1 & S2
    with tf.variable_scope('psi_S1', reuse=tf.AUTO_REUSE) as scope:
    	psi_S1 = tf.layers.conv1d(inputs=S1, filters=out_channels, kernel_size=1, strides=1, padding='valid', data_format='channels_last', activation=activation, name='psi_s1')

    with tf.variable_scope('psi_S2', reuse=tf.AUTO_REUSE) as scope:
    	psi_S2 = tf.layers.conv1d(inputs=S2, filters=out_channels, kernel_size=1, strides=1, padding='valid', data_format='channels_last', activation=activation, name='psi_s2')

    with tf.variable_scope('psi_S3', reuse=tf.AUTO_REUSE) as scope:
    	psi_S3 = tf.layers.conv1d(inputs=S3, filters=out_channels, kernel_size=1, strides=1, padding='valid', data_format='channels_last', activation=activation, name='psi_s2')
    	    	    	    	     
    	    	     
    # Multiply processed state by the attention
    print("psi_S1", psi_S1.shape)
    print("psi_S2", psi_S2.shape)
    print("psi_S3", psi_S3.shape)
    
    print(" -- only one neighborhood (reshape attention) --")
    att_s1 = tf.squeeze(att_s1, axis=2)
    att_s2 = tf.squeeze(att_s2, axis=2)
    att_s3 = tf.squeeze(att_s3, axis=2)
    

    print("att_s1", att_s1.shape)
    print("att_s2", att_s2.shape)
    print("att_s3", att_s3.shape)
        
    
    #psi_S1 = tf.tile(tf.expand_dims(psi_S1, 2), [1, 1, nsample, 1])
    #psi_S2 = tf.tile(tf.expand_dims(psi_S2, 2), [1, 1, nsample, 1])
    
    S1 = att_s1 * psi_S1
    S2 = att_s2 * psi_S2
    S3 = att_s3 * psi_S3    
    
    print("S1", S1)
    print("S2", S2)
    print("S3", S3)
    print("")
    
    # 4. Aggregation 
    # 4.1 Find point in P1 that match P0
    dist, P1_idx = knn_point(nsample, P1 ,P0) # For each point in P0 find the k-cloest point in P3
    print("P1_idx", P1_idx)
    S1_grouped_to_P0 = group_point(S1, P1_idx)
    print("S1_grouped_to_P0", S1_grouped_to_P0)  
    
    # 4.1 Find point in P2 that match P0
    dist, P2_idx = knn_point(nsample, P2 ,P0) # For each point in P0 find the k-cloest point in P3
    print("P2_idx", P2_idx)
    S2_grouped_to_P0 = group_point(S2, P2_idx)
    print("S2_grouped_to_P0", S2_grouped_to_P0)   

    # 4.1 Find point in P3 that match P0
    dist, P3_idx = knn_point(nsample, P3 ,P0) # For each point in P0 find the k-cloest point in P3
    print("P3_idx", P3_idx)
    S3_grouped_to_P0 = group_point(S3, P3_idx)
    print("S3_grouped_to_P0", S3_grouped_to_P0)
    S3_grouped_to_P0 = tf.squeeze(S3_grouped_to_P0, axis=2)   
    
        
    # Reduce mean    
    #S_agrregation = tf.concat( [S1_grouped_to_P0, S2_grouped_to_P0], axis=2)
    #print("S_agrregation", S_agrregation)    
    #Use mean
    #S_agrregation = tf.reduce_mean(S_agrregation, axis =2)
    #print("S_agrregation", S_agrregation)    
    

    #Do huge concatenation
    S_agrregation = tf.concat( [S1_grouped_to_P0, S2_grouped_to_P0], axis=3)
    print("S_agrregation", S_agrregation)
    #S_agrregation = tf.squeeze(S_agrregation)
    S_agrregation =  tf.reshape(S_agrregation, (S_agrregation.shape[0], S_agrregation.shape[1], S_agrregation.shape[3]) )
    print("S_agrregation", S_agrregation)
    


    # 5. Process S3 and S_agreggation together
    concatention = tf.concat( [S3_grouped_to_P0, S_agrregation], axis=2)
    #print("concatention", concatention)
    with tf.variable_scope('Sf_layer', reuse=tf.AUTO_REUSE) as scope:
    	Sf = tf.layers.conv1d(inputs= concatention, filters=out_channels, kernel_size=1, strides=1, padding='valid', data_format='channels_last', activation=tf.nn.relu, name='final_Sf')    
    
    print("Sf", Sf)
    
    return (Sf, att_s1, att_s2) 


def Efficient_GraphAttention_States_Combination_Full_Attention_perpoint(P0,
              P1,
              P2,
              P3,
              S1,
              S2,
              S3,
              nsample,
              activation,
              out_channels,
              scope='attention_states_combination'):

    """
    Learn Sf for all P0 points
    Input:
        S1:     (batch_size, npoint, 3)
        S2:     (batch_size, npoint, feat_channels)
        S3:     
    Output:
        Sf:     (batch_size, npoint, out_channels)
    """
    print(" ==  Graph Efficient Attention FULL ATTENTION Combination ===")
    
    print("P0", P0)
    print("P1", P1)
    print("P2", P2)
    print("P3", P3)

    print("S1", S1)
    print("S2", S2)
    print("S3", S3)
    print("")
    
    # 1. Mathch point between the point clouds.
    """
    Geo_Adj = tf_util.pairwise_distance_2point_cloud(P3, P1)
    Geo_Adj= Geo_Adj[0]
    P1_idx = tf_util.knn(Geo_Adj, k= nsample)
    print("P1_idx", P1_idx)
    # return for each point in P1 the neighboorhos in P3
    """
    
    
    # Grouping for P1
    # 1.3 Find point in P1 for S3
    dist, idx = knn_point(nsample, P3 ,P1) # For each point in P1 find the k-cloest point in S3
    print("idx - ", idx)
    S3_grouped_to_P1 = group_point(S3, idx)
    print("S3_grouped_to_P1", S3_grouped_to_P1)    


    dist, idx = knn_point(nsample, P2 ,P1) # For each point in P1 find the k-cloest point in S3
    S2_grouped_to_P1 = group_point(S2, idx)
    print("S2_grouped_to_P1", S2_grouped_to_P1)    
    
    
    
    #1.1 Expand points S3 to match dimension
    S1_expanded = tf.tile(tf.expand_dims(S1, 2), [1, 1, nsample, 1])
    print("S1_expanded", S1_expanded)    
    S2_expanded = tf.tile(tf.expand_dims(S2, 2), [1, 1, nsample, 1])
    print("S2_expanded", S2_expanded)    	
    S3_expanded = tf.tile(tf.expand_dims(S3, 2), [1, 1, nsample, 1])
    print("S3_expanded", S2_expanded)    
    print("")
        
    
    
    
    
    # 2. Learn Attention Weights
    print(" -- Learn Attention --")
    #For states 1
    concatenation = tf.concat( [S3_grouped_to_P1,S2_grouped_to_P1,S1_expanded], axis = 3)
    print("concatenation", concatenation)
    with tf.variable_scope('attention_s1', reuse=tf.AUTO_REUSE) as scope:
    	att_s1 = tf.layers.conv2d(inputs=concatenation, filters=1, kernel_size=1, strides=1, padding='valid', data_format='channels_last', activation=activation, name='at_s1')
    
    #For states 2
    concatenation = tf.concat( [S3_grouped_to_P1,S2_grouped_to_P1,S1_expanded], axis = 3)
    print("concatenation", concatenation)
    with tf.variable_scope('attention_s2', reuse=tf.AUTO_REUSE) as scope:
    	att_s2 = tf.layers.conv2d(inputs=concatenation, filters=1, kernel_size=1, strides=1, padding='valid', data_format='channels_last', activation=activation, name='at_s2')


    #For states 3
    concatenation = tf.concat( [S3_grouped_to_P1,S2_grouped_to_P1,S1_expanded], axis = 3)
    print("concatenation", concatenation)
    with tf.variable_scope('attention_s3', reuse=tf.AUTO_REUSE) as scope:
    	att_s3 = tf.layers.conv2d(inputs=concatenation, filters=1, kernel_size=1, strides=1, padding='valid', data_format='channels_last', activation=activation, name='at_s2') 

    print("att_s1", att_s1.shape)
    print("att_s2", att_s2.shape)
    print("att_s3", att_s3.shape)
            
    
    # 3 .Process States S1 & S2
    with tf.variable_scope('psi_S1', reuse=tf.AUTO_REUSE) as scope:
    	psi_S1 = tf.layers.conv1d(inputs=S1, filters=out_channels, kernel_size=1, strides=1, padding='valid', data_format='channels_last', activation=activation, name='psi_s1')

    with tf.variable_scope('psi_S2', reuse=tf.AUTO_REUSE) as scope:
    	psi_S2 = tf.layers.conv1d(inputs=S2, filters=out_channels, kernel_size=1, strides=1, padding='valid', data_format='channels_last', activation=activation, name='psi_s2')

    with tf.variable_scope('psi_S3', reuse=tf.AUTO_REUSE) as scope:
    	psi_S3 = tf.layers.conv1d(inputs=S3, filters=out_channels, kernel_size=1, strides=1, padding='valid', data_format='channels_last', activation=activation, name='psi_s2')
    	    	    	    	     
    	    	     
    # Multiply processed state by the attention
    print("psi_S1", psi_S1.shape)
    print("psi_S2", psi_S2.shape)
    print("psi_S3", psi_S3.shape)
    
    print(" -- only one neighborhood (reshape attention) --")
    att_s1 = tf.squeeze(att_s1, axis=2)
    att_s2 = tf.squeeze(att_s2, axis=2)
    att_s3 = tf.squeeze(att_s3, axis=2)
    
    
    print("att_s1", att_s1.shape)
    print("att_s2", att_s2.shape)
    print("att_s3", att_s3.shape)
        
 
    print(" -- Reshape pshi  --")

    # Grouping for P1
    # 1.3 Find point in P1 for S3
    dist, idx = knn_point(nsample, P3 ,P1) # For each point in P1 find the k-cloest point in S3
    print("idx - ", idx)
    S3_grouped_to_P1 = group_point(psi_S3, idx)
    print("S3_grouped_to_P1", S3_grouped_to_P1)    


    dist, idx = knn_point(nsample, P2 ,P1) # For each point in P1 find the k-cloest point in S3
    S2_grouped_to_P1 = group_point(psi_S2, idx)
    print("S2_grouped_to_P1", S2_grouped_to_P1)    
    

    S2_grouped_to_P1 = tf.squeeze(S2_grouped_to_P1, axis=2)
    S3_grouped_to_P1 = tf.squeeze(S3_grouped_to_P1, axis=2)
    
    print("S2_grouped_to_P1", S2_grouped_to_P1)
    print("S3_grouped_to_P1", S3_grouped_to_P1)
    
    S1 = att_s1 * psi_S1 *0
    S2 = att_s2 * S2_grouped_to_P1 *0
    S3 = att_s3 * S3_grouped_to_P1 *1   
    

    print("S1", S1)
    print("S2", S2)
    print("S3", S3)
    print("")
    
        

    # 5. Process S3 and S_agreggation together
    concatention = tf.concat( [S1, S2, S3], axis=2)
    print("concatenation", concatenation)
    #print("concatention", concatention)
    with tf.variable_scope('Sf_layer', reuse=tf.AUTO_REUSE) as scope:
    	Sf = tf.layers.conv1d(inputs= concatention, filters=out_channels, kernel_size=1, strides=1, padding='valid', data_format='channels_last', activation=tf.nn.relu, name='final_Sf')    
    
    print("Sf", Sf)

         
    
    # 4. Aggregation 
    # 4.1 Find point in P1 that match P0
    dist, idx = knn_point(nsample, P1 ,P0) # For each point in P0 find the k-cloest point in P3
    print("idx", idx)
    Sf_grouped_to_P0 = group_point(Sf, idx)
    
    Sf_grouped_to_P0 = tf.squeeze(Sf_grouped_to_P0, axis=2)   
    Sf = Sf_grouped_to_P0
    print("Sf", Sf)
    
    
    return (Sf, att_s1, att_s2, att_s3, psi_S1, S2_grouped_to_P1, S3_grouped_to_P1) 

def Efficient_GraphAttention_States_Combination_Full_Attention_perpoint_prerefined(P0,
              P1,
              P2,
              P3,
              S1,
              S2,
              S3,
              nsample,
              activation,
              out_channels,
              scope='attention_states_combination'):

    """
    Learn Sf for all P0 points
    Input:
        S1:     (batch_size, npoint, 3)
        S2:     (batch_size, npoint, feat_channels)
        S3:     
    Output:
        Sf:     (batch_size, npoint, out_channels)
    """
    print(" ==  Attention Pre-Refined ===")
    
    print("P0", P0)
    print("P1", P1)
    print("P2", P2)
    print("P3", P3)

    print("S1", S1)
    print("S2", S2)
    print("S3", S3)
    print("")
    
    # 1. Mathch point between the point clouds.
    """
    Geo_Adj = tf_util.pairwise_distance_2point_cloud(P3, P1)
    Geo_Adj= Geo_Adj[0]
    P1_idx = tf_util.knn(Geo_Adj, k= nsample)
    print("P1_idx", P1_idx)
    # return for each point in P1 the neighboorhos in P3
    """
    
    
    # Refined ALL Feature
    # 3 .Process States S1 & S2
    with tf.variable_scope('psi_S1', reuse=tf.AUTO_REUSE) as scope:
    	psi_S1 = tf.layers.conv1d(inputs=S1, filters=out_channels, kernel_size=1, strides=1, padding='valid', data_format='channels_last', activation=tf.nn.relu, name='psi_s1')

    with tf.variable_scope('psi_S2', reuse=tf.AUTO_REUSE) as scope:
    	psi_S2 = tf.layers.conv1d(inputs=S2, filters=out_channels, kernel_size=1, strides=1, padding='valid', data_format='channels_last', activation=tf.nn.relu, name='psi_s2')

    with tf.variable_scope('psi_S3', reuse=tf.AUTO_REUSE) as scope:
    	psi_S3 = tf.layers.conv1d(inputs=S3, filters=out_channels, kernel_size=1, strides=1, padding='valid', data_format='channels_last', activation=tf.nn.relu, name='psi_s2')
    	    	    	    	     
    	    	     
    # Multiply processed state by the attention
    print("psi_S1", psi_S1.shape)
    print("psi_S2", psi_S2.shape)
    print("psi_S3", psi_S3.shape)

    # Grouping for P1
    # 1.3 Find point in P1 for S3
    dist, idx = knn_point(nsample, P3 ,P1) # For each point in P1 find the k-cloest point in S3
    print("idx - ", idx)
    S3_grouped_to_P1 = group_point(psi_S3, idx)
    print("S3_grouped_to_P1", S3_grouped_to_P1)    
    dist, idx = knn_point(nsample, P2 ,P1) # For each point in P1 find the k-cloest point in S3
    S2_grouped_to_P1 = group_point(psi_S2, idx)
    print("S2_grouped_to_P1", S2_grouped_to_P1)    
    

    
    #1.1 Expand points S3 to match dimension
    S1_expanded = tf.tile(tf.expand_dims(psi_S1, 2), [1, 1, nsample, 1])
    print("S1_expanded", S1_expanded)    
    S2_expanded = tf.tile(tf.expand_dims(psi_S2, 2), [1, 1, nsample, 1])
    print("S2_expanded", S2_expanded)    	
    S3_expanded = tf.tile(tf.expand_dims(psi_S3, 2), [1, 1, nsample, 1])
    print("S3_expanded", S2_expanded)    
    print("")
        
    
    
 
    # 2. Learn Attention Weights
    print(" -- Learn Attention --")
    #For states 1
    concatenation = tf.concat( [S3_grouped_to_P1,S2_grouped_to_P1,S1_expanded], axis = 3)
    print("concatenation", concatenation)
    with tf.variable_scope('attention_s1', reuse=tf.AUTO_REUSE) as scope:
    	att_s1 = tf.layers.conv2d(inputs=concatenation, filters=1, kernel_size=1, strides=1, padding='valid', data_format='channels_last', activation=activation, name='at_s1')
    #For states 2
    with tf.variable_scope('attention_s2', reuse=tf.AUTO_REUSE) as scope:
    	att_s2 = tf.layers.conv2d(inputs=concatenation, filters=1, kernel_size=1, strides=1, padding='valid', data_format='channels_last', activation=activation, name='at_s2')
    #For states 3
    with tf.variable_scope('attention_s3', reuse=tf.AUTO_REUSE) as scope:
    	att_s3 = tf.layers.conv2d(inputs=concatenation, filters=1, kernel_size=1, strides=1, padding='valid', data_format='channels_last', activation=activation, name='at_s2') 

    print("att_s1", att_s1.shape)
    print("att_s2", att_s2.shape)
    print("att_s3", att_s3.shape)
            
    	    	     
    	    	     
    # Multiply processed state by the attention
    att_s1 = tf.squeeze(att_s1, axis=2)
    att_s2 = tf.squeeze(att_s2, axis=2)
    att_s3 = tf.squeeze(att_s3, axis=2)
    

    print("att_s1", att_s1.shape)
    print("att_s2", att_s2.shape)
    print("att_s3", att_s3.shape)
        
    

    # Reshape 
    S2_grouped_to_P1 = tf.squeeze(S2_grouped_to_P1, axis=2)
    S3_grouped_to_P1 = tf.squeeze(S3_grouped_to_P1, axis=2)
    
    print("S2_grouped_to_P1", S2_grouped_to_P1)
    print("S3_grouped_to_P1", S3_grouped_to_P1)
    
    S1 = att_s1 * psi_S1 *1
    S2 = att_s2 * S2_grouped_to_P1 *1
    S3 = att_s3 * S3_grouped_to_P1 *1   
    

    print("S1", S1)
    print("S2", S2)
    print("S3", S3)
    print("")
    
        

    # 5. Process S3 and S_agreggation together
    concatention = tf.concat( [S1, S2, S3], axis=2)
    print("concatenation", concatenation)
    #print("concatention", concatention)
    with tf.variable_scope('Sf_layer', reuse=tf.AUTO_REUSE) as scope:
    	Sf = tf.layers.conv1d(inputs= concatention, filters=out_channels, kernel_size=1, strides=1, padding='valid', data_format='channels_last', activation=tf.nn.relu, name='final_Sf')    
    
    print("Sf", Sf)

         
    # 4. Aggregation 
    # 4.1 Find point in P1 that match P0
    dist, idx = knn_point(nsample, P1 ,P0) # For each point in P0 find the k-cloest point in P3
    print("idx", idx)
    Sf_grouped_to_P0 = group_point(Sf, idx)
    
    Sf_grouped_to_P0 = tf.squeeze(Sf_grouped_to_P0, axis=2)   
    Sf = Sf_grouped_to_P0
    print("Sf", Sf)
    
    
    return (Sf, att_s1, att_s2, att_s3, psi_S1, S2_grouped_to_P1, S3_grouped_to_P1) 

def Adaptative_Extra_Attention(P0,
              P1,
              P2,
              P3,
              S1,
              S2,
              S3,
              nsample,
              activation,
              out_channels,
              scope='attention_states_combination'):

    """
    Learn Sf for all P0 points
    Input:
        S1:     (batch_size, npoint, 3)
        S2:     (batch_size, npoint, feat_channels)
        S3:     
    Output:
        Sf:     (batch_size, npoint, out_channels)
    """
    print(" ==  Attention Pre-Refined EXTRA ATTENTION===")
    
    print("P0", P0)
    print("P1", P1)
    print("P2", P2)
    print("P3", P3)

    print("S1", S1)
    print("S2", S2)
    print("S3", S3)
    print("")
    
    # 1. Mathch point between the point clouds.
    """
    Geo_Adj = tf_util.pairwise_distance_2point_cloud(P3, P1)
    Geo_Adj= Geo_Adj[0]
    P1_idx = tf_util.knn(Geo_Adj, k= nsample)
    print("P1_idx", P1_idx)
    # return for each point in P1 the neighboorhos in P3
    """
    
    
    # Refined ALL Feature
    # 3 .Process States S1 & S2
    with tf.variable_scope('psi_S1', reuse=tf.AUTO_REUSE) as scope:
    	psi_S1 = tf.layers.conv1d(inputs=S1, filters=out_channels, kernel_size=1, strides=1, padding='valid', data_format='channels_last', activation=tf.nn.relu, name='psi_s1')

    with tf.variable_scope('psi_S2', reuse=tf.AUTO_REUSE) as scope:
    	psi_S2 = tf.layers.conv1d(inputs=S2, filters=out_channels, kernel_size=1, strides=1, padding='valid', data_format='channels_last', activation=tf.nn.relu, name='psi_s2')

    with tf.variable_scope('psi_S3', reuse=tf.AUTO_REUSE) as scope:
    	psi_S3 = tf.layers.conv1d(inputs=S3, filters=out_channels, kernel_size=1, strides=1, padding='valid', data_format='channels_last', activation=tf.nn.relu, name='psi_s2')
    	    	    	    	     
    	    	     
    # Multiply processed state by the attention
    print("psi_S1", psi_S1.shape)
    print("psi_S2", psi_S2.shape)
    print("psi_S3", psi_S3.shape)

    # Grouping for P1
    # 1.3 Find point in P1 for S3
    dist, idx = knn_point(nsample, P3 ,P1) # For each point in P1 find the k-cloest point in S3
    print("idx - ", idx)
    S3_grouped_to_P1 = group_point(psi_S3, idx)
    print("S3_grouped_to_P1", S3_grouped_to_P1)    
    dist, idx = knn_point(nsample, P2 ,P1) # For each point in P1 find the k-cloest point in S3
    S2_grouped_to_P1 = group_point(psi_S2, idx)
    print("S2_grouped_to_P1", S2_grouped_to_P1)    
    

    
    #1.1 Expand points S3 to match dimension
    S1_expanded = tf.tile(tf.expand_dims(psi_S1, 2), [1, 1, nsample, 1])
    print("S1_expanded", S1_expanded)    
    S2_expanded = tf.tile(tf.expand_dims(psi_S2, 2), [1, 1, nsample, 1])
    print("S2_expanded", S2_expanded)    	
    S3_expanded = tf.tile(tf.expand_dims(psi_S3, 2), [1, 1, nsample, 1])
    print("S3_expanded", S2_expanded)    
    print("")
        
    
    
 
    # 2. Learn Attention Weights
    print(" -- Learn Attention --")
    #For states 1
    concatenation = tf.concat( [S3_grouped_to_P1,S2_grouped_to_P1,S1_expanded], axis = 3)
    print("concatenation", concatenation)
    with tf.variable_scope('attention_s1', reuse=tf.AUTO_REUSE) as scope:
    	b_att_s1 = tf.layers.conv2d(inputs=concatenation, filters=64, kernel_size=1, strides=1, padding='valid', data_format='channels_last', activation=tf.nn.relu, name='at_s1')
    #For states 2
    with tf.variable_scope('attention_s2', reuse=tf.AUTO_REUSE) as scope:
    	b_att_s2 = tf.layers.conv2d(inputs=concatenation, filters=64, kernel_size=1, strides=1, padding='valid', data_format='channels_last', activation=tf.nn.relu, name='at_s2')
    #For states 3
    with tf.variable_scope('attention_s3', reuse=tf.AUTO_REUSE) as scope:
    	b_att_s3 = tf.layers.conv2d(inputs=concatenation, filters=64, kernel_size=1, strides=1, padding='valid', data_format='channels_last', activation=tf.nn.relu, name='at_s2') 


    with tf.variable_scope('extra_attention_s1', reuse=tf.AUTO_REUSE) as scope:
    	att_s1 = tf.layers.conv2d(inputs=b_att_s1, filters=1, kernel_size=1, strides=1, padding='valid', data_format='channels_last', activation=None, name='at_s1')
    #For states 2
    with tf.variable_scope('extra_attention_s2', reuse=tf.AUTO_REUSE) as scope:
    	att_s2 = tf.layers.conv2d(inputs=b_att_s2, filters=1, kernel_size=1, strides=1, padding='valid', data_format='channels_last', activation=None, name='at_s2')
    #For states 3
    with tf.variable_scope('extra_attention_s3', reuse=tf.AUTO_REUSE) as scope:
    	att_s3 = tf.layers.conv2d(inputs=b_att_s3, filters=1, kernel_size=1, strides=1, padding='valid', data_format='channels_last', activation=None, name='at_s2') 
    	
    	

    print("att_s1", att_s1.shape)
    print("att_s2", att_s2.shape)
    print("att_s3", att_s3.shape)
            
    	    	     
    	    	     
    # Multiply processed state by the attention
    att_s1 = tf.squeeze(att_s1, axis=2)
    att_s2 = tf.squeeze(att_s2, axis=2)
    att_s3 = tf.squeeze(att_s3, axis=2)
    

    print("att_s1", att_s1.shape)
    print("att_s2", att_s2.shape)
    print("att_s3", att_s3.shape)
        
    

    # Reshape 
    S2_grouped_to_P1 = tf.squeeze(S2_grouped_to_P1, axis=2)
    S3_grouped_to_P1 = tf.squeeze(S3_grouped_to_P1, axis=2)
    
    print("S2_grouped_to_P1", S2_grouped_to_P1)
    print("S3_grouped_to_P1", S3_grouped_to_P1)
    
    S1 = att_s1 * psi_S1 *1
    S2 = att_s2 * S2_grouped_to_P1 *1
    S3 = att_s3 * S3_grouped_to_P1 *1   
    

    print("S1", S1)
    print("S2", S2)
    print("S3", S3)
    print("")
    
        

    # 5. Process S3 and S_agreggation together
    concatention = tf.concat( [S1, S2, S3], axis=2)
    print("concatenation", concatenation)
    #print("concatention", concatention)
    with tf.variable_scope('Sf_layer', reuse=tf.AUTO_REUSE) as scope:
    	Sf = tf.layers.conv1d(inputs= concatention, filters=out_channels, kernel_size=1, strides=1, padding='valid', data_format='channels_last', activation=tf.nn.relu, name='final_Sf')    
    
    print("Sf", Sf)

         
    # 4. Aggregation 
    # 4.1 Find point in P1 that match P0
    dist, idx = knn_point(nsample, P1 ,P0) # For each point in P0 find the k-cloest point in P3
    print("idx", idx)
    Sf_grouped_to_P0 = group_point(Sf, idx)
    
    Sf_grouped_to_P0 = tf.squeeze(Sf_grouped_to_P0, axis=2)   
    Sf = Sf_grouped_to_P0
    print("Sf", Sf)
    
    
    return (Sf, att_s1, att_s2, att_s3, psi_S1, S2_grouped_to_P1, S3_grouped_to_P1) 


          

