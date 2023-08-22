"""
PointNet++ Operations and Layers
Original Author: Charles R. Qi
Modified by Hehe Fan
Data September 2019
"""

import os
import sys
import numpy as np
import tensorflow as tf

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, "modules/tf_ops/sampling"))
sys.path.append(os.path.join(ROOT_DIR, "modules/tf_ops/grouping"))
sys.path.append(os.path.join(ROOT_DIR, "modules/tf_ops/3d_interpolation"))
from tf_sampling import farthest_point_sample, gather_point
from tf_grouping import query_ball_point, group_point, knn_point
from tf_interpolate import three_nn, three_interpolate


def sample_and_group(
    npoint, radius, nsample, xyz, color, features, states, knn=False, use_xyz=True
):
    """
    Input:
        npoint:         int32
        radius:         float32
        nsample:        int32
        xyz:            (batch_size, ndataset, 3) TF tensor
        features:       (batch_size, ndataset, channel) TF tensor, if None will just use xyz as features
        color:          (batch_size, ndataset, 3) TF tensor, if None will just use xyz as features
        knn:            bool, if True use kNN instead of radius search
        use_xyz:        bool, if True concat XYZ with local point features, otherwise just use point features
    Output:
        new_xyz:        (batch_size, npoint, 3) TF tensor
        new_points:     (batch_size, npoint, nsample, 3+channel) TF tensor
        idx:            (batch_size, npoint, nsample) TF tensor, indices of local points as in ndataset points
        grouped_xyz:    (batch_size, npoint, nsample, 3) TF tensor, normalized point XYZs (subtracted by seed point XYZ) in local regions
    """

    # Group Points
    FPS_index = farthest_point_sample(npoint, xyz)
    new_xyz = gather_point(xyz, FPS_index)  # (batch_size, npoint, 3)
    new_color = gather_point(color, FPS_index)  # (batch_size, npoint, 3)

    if knn:
        _, idx = knn_point(nsample, xyz, new_xyz)
    else:
        idx, pts_cnt = query_ball_point(radius, nsample, xyz, new_xyz)

    grouped_xyz = group_point(xyz, idx)  # (batch_size, npoint, nsample, 3)
    grouped_xyz -= tf.tile(
        tf.expand_dims(new_xyz, 2), [1, 1, nsample, 1]
    )  # translation normalization

    print("FPS_index", FPS_index)
    print("idx", FPS_index)
    # Group Features
    if features is not None:
        grouped_features = group_point(
            features, idx
        )  # (batch_size, npoint, nsample, channel)
        if use_xyz:
            new_features = tf.concat(
                [grouped_xyz, grouped_features], axis=-1
            )  # (batch_size, npoint, nsample, 3+channel)
        else:
            new_features = grouped_features
            print("grouped_features", grouped_features)
    else:
        new_features = grouped_xyz

    # Group States
    if states is not None:
        grouped_states = group_point(
            states, idx
        )  # (batch_size, npoint, nsample, channel)
        if use_xyz:
            new_states = tf.concat(
                [grouped_xyz, grouped_states], axis=-1
            )  # (batch_size, npoint, nsample, 3+channel)
        else:
            new_states = grouped_states
    else:
        new_states = grouped_xyz

    return new_xyz, new_color, new_features, new_states, idx, grouped_xyz


def sample_and_group_all(xyz, points, use_xyz=True):
    """
    Inputs:
        xyz:        (batch_size, ndataset, 3) TF tensor
        points:     (batch_size, ndataset, channel) TF tensor, if None will just use xyz as points
        use_xyz:    bool, if True concat XYZ with local point features, otherwise just use point features
    Outputs:
        new_xyz:    (batch_size, 1, 3) as (0,0,0)
        new_points: (batch_size, 1, ndataset, 3+channel) TF tensor
    Note:
        Equivalent to sample_and_group with npoint=1, radius=inf, use (0,0,0) as the centroid
    """
    batch_size = xyz.get_shape()[0].value
    nsample = xyz.get_shape()[1].value

    new_xyz = tf.constant(
        np.tile(np.array([0, 0, 0]).reshape((1, 1, 3)), (batch_size, 1, 1)),
        dtype=tf.float32,
    )  # (batch_size, 1, 3)
    idx = tf.constant(
        np.tile(np.array(range(nsample)).reshape((1, 1, nsample)), (batch_size, 1, 1))
    )
    grouped_xyz = tf.reshape(
        xyz, (batch_size, 1, nsample, 3)
    )  # (batch_size, npoint=1, nsample, 3)

    if points is not None:
        if use_xyz:
            new_points = tf.concat([xyz, points], axis=2)  # (batch_size, 16, 259)
        else:
            new_points = points
        new_points = tf.expand_dims(new_points, 1)  # (batch_size, 1, 16, 259)
    else:
        new_points = grouped_xyz

    return new_xyz, new_points, idx, grouped_xyz


def conv2d(
    inputs,
    filters,
    kernel_size=[1, 1],
    strides=(1, 1),
    padding="valid",
    data_format="channels_last",
    activation=tf.nn.relu,
    name="conv2d",
):
    outputs = tf.layers.conv2d(
        inputs=inputs,
        filters=filters,
        kernel_size=kernel_size,
        strides=strides,
        padding=padding,
        data_format=data_format,
        activation=activation,
        name=name + "_conv2d",
    )
    return outputs


def pointnet_sa_module(
    xyz,
    points,
    npoint,
    radius,
    nsample,
    mlp,
    mlp2=None,
    group_all=False,
    knn=False,
    pooling="max",
    use_xyz=True,
    scope="sa",
):
    """PointNet Set Abstraction (SA) Module
    Input:
        xyz:        (batch_size, ndataset, 3) TF tensor
        points:     (batch_size, ndataset, channel) TF tensor
        npoint:     int32 -- #points sampled in farthest point sampling
        radius:     float32 -- search radius in local region
        nsample:    int32 -- how many points in each local region
        mlp:        list of int32 -- output size for MLP on each point
        mlp2:       list of int32 -- output size for MLP on each region
        group_all:  bool -- group all points into one PC if set true, OVERRIDE npoint, radius and nsample settings
        use_xyz:    bool, if True concat XYZ with local point features, otherwise just use point features
    Return:
        new_xyz:    (batch_size, npoint, 3) TF tensor
        new_points: (batch_size, npoint, mlp[-1] or mlp2[-1]) TF tensor
        idx:        (batch_size, npoint, nsample) int32 -- indices for local regions
    """

    with tf.variable_scope(scope) as sc:
        # Sample and Grouping
        if group_all:
            nsample = xyz.get_shape()[1].value
            new_xyz, new_points, idx, grouped_xyz = sample_and_group_all(
                xyz, points, use_xyz
            )
        else:
            new_xyz, new_points, idx, grouped_xyz = sample_and_group(
                npoint, radius, nsample, xyz, points, knn, use_xyz
            )

        # Point Feature Embedding
        for i, num_out_channel in enumerate(mlp):
            new_points = conv2d(
                inputs=new_points, filters=num_out_channel, name="mlp_%d" % (i + 1)
            )

        # Pooling in Local Regions
        if pooling == "max":
            new_points = tf.reduce_max(
                new_points, axis=[2], keepdims=True, name="maxpool"
            )
        elif pooling == "avg":
            new_points = tf.reduce_mean(
                new_points, axis=[2], keepdims=True, name="avgpool"
            )
        elif pooling == "weighted_avg":
            with tf.variable_scope("weighted_avg"):
                dists = tf.norm(grouped_xyz, axis=-1, ord=2, keepdims=True)
                exp_dists = tf.exp(-dists * 5)
                weights = exp_dists / tf.reduce_sum(
                    exp_dists, axis=2, keepdims=True
                )  # (batch_size, npoint, nsample, 1)
                new_points *= weights  # (batch_size, npoint, nsample, mlp[-1])
                new_points = tf.reduce_sum(new_points, axis=2, keepdims=True)
        elif pooling == "max_and_avg":
            max_points = tf.reduce_max(
                new_points, axis=[2], keepdims=True, name="maxpool"
            )
            avg_points = tf.reduce_mean(
                new_points, axis=[2], keepdims=True, name="avgpool"
            )
            new_points = tf.concat([avg_points, max_points], axis=-1)

        # [Optional] Further Processing
        if mlp2 is not None:
            for i, num_out_channel in enumerate(mlp2):
                new_points = conv2d(
                    inputs=new_points, filters=num_out_channel, name="mlp2_%d" % (i + 1)
                )

        new_points = tf.squeeze(new_points, [2])  # (batch_size, npoints, mlp2[-1])

        return new_xyz, new_points, idx


def pointnet_fp_module(
    xyz1, xyz2, feat1, feat2, mlp, last_mlp_activation=True, scope="fp"
):
    """PointNet Feature Propogation (FP) Module
    Input:
        xyz1:       (batch_size, ndataset1, 3) TF tensor
        xyz2:       (batch_size, ndataset2, 3) TF tensor, sparser than xyz1
        feat1:    (batch_size, ndataset1, nchannel1) TF tensor
        points2:    (batch_size, ndataset2, nchannel2) TF tensor
        mlp:        list of int32 -- output size for MLP on each point
    Return:
        new_points: (batch_size, ndataset1, mlp[-1]) TF tensor
    """
    # print("\n State Propagation only Interpolation")

    with tf.variable_scope(scope) as sc:
        dist, idx = three_nn(xyz1, xyz2)
        dist = tf.maximum(dist, 1e-10)
        norm = tf.reduce_sum((1.0 / dist), axis=2, keepdims=True)
        norm = tf.tile(norm, [1, 1, 3])
        weight = (1.0 / dist) / norm
        interpolated_points = three_interpolate(feat2, idx, weight)
        print("interpolated_points", interpolated_points)

        if feat1 is not None:
            new_feat1 = tf.concat(
                axis=2, values=[interpolated_points, feat1]
            )  # B,ndataset1,nchannel1+nchannel2
        else:
            new_feat1 = interpolated_points
        new_feat1 = tf.expand_dims(new_feat1, 2)
        for i, num_out_channel in enumerate(mlp):
            if i == len(mlp) - 1 and not (last_mlp_activation):
                activation_fn = None
            else:
                activation_fn = tf.nn.relu
            new_feat1 = conv2d(
                inputs=new_feat1, filters=num_out_channel, name="mlp_%d" % (i + 1)
            )

        # print("[1] new_feat1",new_feat1)
        new_feat1 = tf.squeeze(new_feat1, [2])  # B,ndataset1,mlp[-1]

        # print("[2] new_feat1",new_feat1)
        # new_feat1 = interpolated_points

        return new_feat1


def pointnet_fp_module_original(
    xyz1, xyz2, points1, points2, mlp, last_mlp_activation=True, scope="fp"
):
    """PointNet Feature Propogation (FP) Module
    Input:
        xyz1:       (batch_size, ndataset1, 3) TF tensor
        xyz2:       (batch_size, ndataset2, 3) TF tensor, sparser than xyz1
        points1:    (batch_size, ndataset1, nchannel1) TF tensor
        points2:    (batch_size, ndataset2, nchannel2) TF tensor
        mlp:        list of int32 -- output size for MLP on each point
    Return:
        new_points: (batch_size, ndataset1, mlp[-1]) TF tensor
    """
    print("Original State Propgation")

    with tf.variable_scope(scope) as sc:
        dist, idx = three_nn(xyz1, xyz2)
        dist = tf.maximum(dist, 1e-10)
        norm = tf.reduce_sum((1.0 / dist), axis=2, keepdims=True)
        norm = tf.tile(norm, [1, 1, 3])
        weight = (1.0 / dist) / norm
        interpolated_points = three_interpolate(points2, idx, weight)

        if points1 is not None:
            new_points1 = tf.concat(
                axis=2, values=[interpolated_points, points1]
            )  # B,ndataset1,nchannel1+nchannel2
        else:
            new_points1 = interpolated_points
        new_points1 = tf.expand_dims(new_points1, 2)
        for i, num_out_channel in enumerate(mlp):
            if i == len(mlp) - 1 and not (last_mlp_activation):
                activation_fn = None
            else:
                activation_fn = tf.nn.relu
            new_points1 = conv2d(
                inputs=new_points1, filters=num_out_channel, name="mlp_%d" % (i + 1)
            )

        new_points1 = tf.squeeze(new_points1, [2])  # B,ndataset1,mlp[-1]
        return new_points1


""" Return interpolated points as well """


def pointnet_fp_module_original_interpolated(
    xyz1, xyz2, points1, points2, mlp, last_mlp_activation=True, scope="fp"
):
    """PointNet Feature Propogation (FP) Module
    Input:
        xyz1:       (batch_size, ndataset1, 3) TF tensor
        xyz2:       (batch_size, ndataset2, 3) TF tensor, sparser than xyz1
        points1:    (batch_size, ndataset1, nchannel1) TF tensor
        points2:    (batch_size, ndataset2, nchannel2) TF tensor
        mlp:        list of int32 -- output size for MLP on each point
    Return:
        new_points: (batch_size, ndataset1, mlp[-1]) TF tensor
    """
    print("Original State Propgation")

    with tf.variable_scope(scope) as sc:
        dist, idx = three_nn(xyz1, xyz2)
        dist = tf.maximum(dist, 1e-10)
        norm = tf.reduce_sum((1.0 / dist), axis=2, keepdims=True)
        norm = tf.tile(norm, [1, 1, 3])
        weight = (1.0 / dist) / norm
        interpolated_points = three_interpolate(points2, idx, weight)

        if points1 is not None:
            new_points1 = tf.concat(
                axis=2, values=[interpolated_points, points1]
            )  # B,ndataset1,nchannel1+nchannel2
        else:
            new_points1 = interpolated_points
        new_points1 = tf.expand_dims(new_points1, 2)
        for i, num_out_channel in enumerate(mlp):
            if i == len(mlp) - 1 and not (last_mlp_activation):
                activation_fn = None
            else:
                activation_fn = tf.nn.relu
            new_points1 = conv2d(
                inputs=new_points1, filters=num_out_channel, name="mlp_%d" % (i + 1)
            )

        new_points1 = tf.squeeze(new_points1, [2])  # B,ndataset1,mlp[-1]

        return new_points1, interpolated_points


def pointnet_interpolation(xyz1, xyz2, points2):
    dist, idx = three_nn(xyz1, xyz2)
    dist = tf.maximum(dist, 1e-10)
    norm = tf.reduce_sum((1.0 / dist), axis=2, keepdims=True)
    norm = tf.tile(norm, [1, 1, 3])
    weight = (1.0 / dist) / norm

    interpolated_points = three_interpolate(points2, idx, weight)

    return interpolated_points


""" Interpolate only the features """
"""Test Functions"""


def copy_feat_testt(xyz1, xyz2, feat2):
    print("\n Upsmaple features Module")
    """
	Input:
	  xyz1: 4000 points
	  xyz2: 1000 points
	  feat: 1000 feat
	Output:
	 interpolated feat: 4000 feat
	"""

    k = 4
    dist, idx = knn_point(k, xyz2, xyz1)
    feat_grouped = group_point(feat2, idx)
    xyz2_grouped = group_point(xyz2, idx)

    # xyz1_expanded = tf.expand_dims(xyz1, 2)
    # displament
    # displacement = xyz2_grouped - xyz1_expanded

    # dist = tf.norm(displacement, ord='euclidean', axis=3, keepdims=True, name=None)
    # dist= tf.nn.softmax(dist)
    # print("dist", dist)
    # print("feat_grouped", feat_grouped)

    # weighted_feat_grouped = feat_grouped *dist
    # print("weighted_feat_grouped", weighted_feat_grouped)

    # feat_grouped= tf.reduce_mean(feat_grouped, axis=[2], keepdims=False)
    interpolated_feat = feat_grouped  # * (1/dist)
    # interpolated_feat = weighted_feat_grouped #* (1/dist)
    interpolated_feat = tf.reduce_mean(interpolated_feat, axis=[2], keepdims=False)

    return interpolated_feat


def copy_feat_test2(xyz1, xyz2, feat2):
    print("\n Upsmaple features Module")
    """
	Input:
	  xyz1: 4000 points
	  xyz2: 1000 points
	  feat2: 1000 feat
	Output:
	 interpolated feat: 4000 feat
	"""

    k = 4
    dist, idx = knn_point(k, xyz2, xyz1)
    feat_grouped = group_point(feat2, idx)
    xyz2_grouped = group_point(xyz2, idx)

    xyz1_expanded = tf.expand_dims(xyz1, 2)
    # displament
    displacement = xyz2_grouped - xyz1_expanded

    dist = tf.norm(displacement, ord="euclidean", axis=3, keepdims=True, name=None)
    dist = tf.nn.softmax(dist)
    print("dist", dist)
    print("feat_grouped", feat_grouped)

    weighted_feat_grouped = feat_grouped * dist
    print("weighted_feat_grouped", weighted_feat_grouped)

    # feat_grouped= tf.reduce_mean(feat_grouped, axis=[2], keepdims=False)
    # interpolated_feat = feat_grouped #* (1/dist)
    interpolated_feat = weighted_feat_grouped  # * (1/dist)
    interpolated_feat = tf.reduce_mean(interpolated_feat, axis=[2], keepdims=False)

    return interpolated_feat


""" Interpolate only the features """


def pointnet_states_concatenation(
    states1, states2, mlp, last_mlp_activation=True, scope="fp"
):
    print("\nstates1", states1)
    print("states2", states2)

    if states1 is not None:
        new_points1 = tf.concat(axis=2, values=[states1, states2])
    else:
        new_points1 = states2

    new_points1 = tf.expand_dims(new_points1, 2)
    print("new_points1", new_points1)

    for i, num_out_channel in enumerate(mlp):
        if i == len(mlp) - 1 and not (last_mlp_activation):
            activation_fn = None
        else:
            activation_fn = tf.nn.relu

        new_points1 = conv2d(
            inputs=new_points1, filters=num_out_channel, name="mlp_%d" % (i + 1)
        )

    print("[f] new_points1", new_points1)

    new_points1 = tf.squeeze(new_points1, [2])
    return new_points1


def states_propagated_by_feat(
    xyz1,
    xyz2,
    states1,
    states2,
    feat1,
    feat2,
    mlp,
    last_mlp_activation=True,
    scope="fp",
):
    """PointNet Feature Propogation (FP) Module
    Input:
        xyz1:       (batch_size, ndataset1, 3) TF tensor
        xyz2:       (batch_size, ndataset2, 3) TF tensor, sparser than xyz1
        states1:    (batch_size, ndataset1, nchannel1) TF tensor
        states2:    (batch_size, ndataset2, nchannel2) TF tensor
        feat1:    (batch_size, ndataset1, nchannel1) TF tensor
        feat2:    (batch_size, ndataset2, nchannel2) TF tensor
        mlp:        list of int32 -- output size for MLP on each point
    Return:
        new_states1: (batch_size, ndataset1, mlp[-1]) TF tensor
    """
    print("\n Propgation cell\n")
    print("xyz1", xyz1)
    print("xyz2", xyz2)
    print("feat1", feat1)
    print("feat2", feat2)
    print("states1", states1)
    print("states2", states2)

    k = 4
    if feat1 is not None:
        dist, idx = knn_point(k, feat2, feat1)  # based on features
    else:
        dist, idx = knn_point(k, xyz2, xyz1)  # based on geometry

    print("idx", idx)
    states2_grouped = group_point(states2, idx)

    if states1 is not None:
        states1_expanded = tf.expand_dims(states1, 2)
        print("states2_grouped", states2_grouped)
        displacement = states2_grouped - states1_expanded
        dist = tf.norm(
            (displacement + 1.0e-12), ord="euclidean", axis=3, keepdims=True, name=None
        )
        dist = tf.nn.softmax(dist, axis=2)
        edge_w = 1 / (dist + 1.0e-12)
        print("edge_w:", edge_w)
        interpolated_states = states2_grouped * edge_w

        # Do the Neighbohood mean
        # interpolated_states =tf.reduce_mean(interpolated_states, axis=[2], keepdims=False)

        sum_interpolated_states = tf.reduce_sum(
            interpolated_states, axis=[2], keepdims=False
        )
        sum_edge_w = tf.reduce_sum(edge_w, axis=[2], keepdims=False)
        print("sum_edge_w", sum_edge_w)
        print("sum_interpolated_states", sum_interpolated_states)
        interpolated_states = sum_interpolated_states / sum_edge_w

    else:
        # no concatenation or weighted aggregation
        interpolated_states = states2_grouped
        interpolated_states = tf.reduce_mean(
            interpolated_states, axis=[2], keepdims=False
        )

    if states1 is not None:
        new_states1 = tf.concat(axis=2, values=[interpolated_states, states1])
    else:
        new_states1 = interpolated_states

    new_states1 = tf.expand_dims(new_states1, 2)

    print("\nConcatenation:", new_states1)

    for i, num_out_channel in enumerate(mlp):
        if i == len(mlp) - 1 and not (last_mlp_activation):
            activation_fn = None
        else:
            activation_fn = tf.nn.relu
        new_states1 = conv2d(
            inputs=new_states1, filters=num_out_channel, name="mlp_%d" % (i + 1)
        )
    print("new_states1:", new_states1)
    new_states1 = tf.squeeze(new_states1, [2])
    print("new_states1:", new_states1)

    return new_states1


def my_tf_round(x, decimals=15):
    multiplier = tf.constant(10**decimals, dtype=x.dtype)
    return tf.round(x * multiplier) / multiplier
