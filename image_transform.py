from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import math
import time
import tensorflow as tf
from tensorflow.contrib import slim
import python_utils
import run_utils
import dataset_multires

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('config', "config",
                    "Name of the config file, excluding the .json file extension")

flags.DEFINE_boolean('new_run', True,
                     "Train from scratch (when True) or train from the last checkpoint (when False)")

flags.DEFINE_string('init_run_name', None,
                    "This is the run_name to initialize the weights from. "
                    "If None, weights will be initialized randomly."
                    "This is a single word, without the timestamp.")

flags.DEFINE_string('run_name', "model",
                    "Continue training from run_name. This is a single word, without the timestamp.")
# If not specified, the last run is used (unless new_run is True or no runs are in the runs directory).
# If new_run is True, creates the new run with name equal run_name.

flags.DEFINE_integer('batch_size', 1, "Batch size. Generally set as large as the VRAM can handle.")


def get_output_res(input_res, pool_count):
    """
    This function has to be re-written if the model architecture changes

    :param input_res:
    :param pool_count:
    :return:
    """
    current_res = input_res
    warning_non_zero_remainder = False
    # branch_image
    for i in range(pool_count):
        current_res -= 4  # 2 conv3x3
        current_res, r = divmod(current_res, 2)  # pool
        warning_non_zero_remainder = warning_non_zero_remainder or bool(r)
    current_res -= 4  # 2 conv3x3 of the last layer
    # common_part
    current_res -= 4  # 2 conv3x3
    # branch_disp
    for i in range(pool_count):
        current_res *= 2  # upsample
        current_res -= 4  # 2 conv3x3
    if warning_non_zero_remainder:
        print(
            "WARNING: a pooling operation will result in a non integer res, the network will automatically add padding there. The output of this function is not garanteed to be exact.")
    return int(current_res)


def get_input_res(output_res, pool_count):
    """
    This function has to be re-written if the model architecture changes

    :param output_res:
    :param pool_count:
    :return:
    """
    current_res = output_res
    warning_non_zero_remainder = False
    # branch_disp
    for i in range(pool_count):
        current_res += 4  # 2 conv3x3
        current_res, r = divmod(current_res, 2)  # upsample
        warning_non_zero_remainder = warning_non_zero_remainder or bool(r)
    # common_part
    current_res += 4  # 2 conv3x3
    # branch_image
    current_res += 4  # 2 conv3x3 of the last layer
    for i in range(pool_count):
        current_res *= 2  # pool
        current_res += 4  # 2 conv3x3
    if warning_non_zero_remainder:
        print(
            "WARNING: a pooling operation will result in a non integer res, the network will automatically add padding there. The output of this function is not garanteed to be exact.")
    return int(current_res)


def warp_error(five_param_pred_logits,gt_rotate_theta,gt_scale_x,gt_scale_y,gt_offset_x,gt_offset_y,input_disp_polygon_map,input_gt_polygon_map):

    pred_rotate_theta = five_param_pred_logits[:,0]
    pred_scale_x = five_param_pred_logits[:,1]
    pred_scale_y = five_param_pred_logits[:,2]
    pred_offset_x = five_param_pred_logits[:,3]
    pred_offset_y = five_param_pred_logits[:,4]

    l2_loss = tf.reduce_mean(
        tf.nn.l2_loss(pred_rotate_theta - gt_rotate_theta) \
      + tf.nn.l2_loss(pred_scale_x - gt_scale_x)\
      + tf.nn.l2_loss(pred_scale_y - gt_scale_y)\
      + tf.nn.l2_loss(pred_offset_x - gt_offset_x)\
      + tf.nn.l2_loss(pred_offset_y - gt_offset_y))

    l1_loss = tf.reduce_mean(tf.abs(pred_rotate_theta - gt_rotate_theta) + tf.abs(pred_scale_x - gt_scale_x) + tf.abs(pred_scale_y - gt_scale_y) + tf.abs(pred_offset_x - gt_offset_x) + tf.abs(pred_offset_y - gt_offset_y))


    h10 = tf.cos(pred_rotate_theta)*pred_scale_x
    h11 = tf.sin(pred_rotate_theta)*pred_scale_y
    h12 = pred_offset_x
    h20 = -tf.sin(pred_rotate_theta)*pred_scale_x
    h21 = tf.cos(pred_rotate_theta)*pred_scale_y
    h22 = pred_offset_y
    h30 = tf.constant([0.0])
    h31 = tf.constant([0.0])
    h32 = tf.constant([1.0])

    # print(tf.concat([h10,h11,h12,h20,h21,h22,h30,h31,h32],axis=-1).shape)

    matrix_3_3 = tf.linalg.inv(tf.reshape(tf.concat([h10,h11,h12,h20,h21,h22,h30,h31,h32],axis=-1),shape=(3,3)))
    matrix_1_8 = tf.contrib.image.matrices_to_flat_transforms(matrix_3_3)

    warp_image = tf.contrib.image.transform(input_disp_polygon_map,matrix_1_8,interpolation="BILINEAR",name=None)
    print("warp_image",warp_image.shape)

    # rotate_matrix = tf.reshape(tf.concat([pred_offset_x,pred_offset_y],axis=-1),[1,2])
    # warp_image = tf.contrib.image.translate(input_disp_polygon_map,rotate_matrix,interpolation="BILINEAR",name=None)

    warp_loss = tf.reduce_mean(tf.abs(warp_image - input_gt_polygon_map))
    return  l2_loss,l1_loss,warp_loss


def compute_current_adam_lr(optimizer):
    return optimizer._lr

def variable_summaries(var):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    # with tf.name_scope('summaries'):
    mean = tf.reduce_mean(var)
    tf.summary.scalar('mean', mean)
    # with tf.name_scope('stddev'):
    stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
    tf.summary.scalar('stddev', stddev)
    tf.summary.scalar('max', tf.reduce_max(var))
    tf.summary.scalar('min', tf.reduce_min(var))
    tf.summary.histogram('histogram', var)


def weight_variable(shape, std_factor=1, wd=None):#[3,3,3,32]
    """weight_variable generates a weight variable of a given shape. Adds weight decay if specified"""
    # Initialize using Xavier initializer
    fan_in = 100
    fan_out = 100
    if len(shape) == 4:
        fan_in = shape[0] * shape[1] * shape[2]
        fan_out = shape[3]
    elif len(shape) == 2:
        fan_in = shape[0]
        fan_out = shape[1]
    else:
        print("WARNING: This shape format is not handled! len(shape) = {}".format(len(shape)))
    stddev = std_factor * math.sqrt(2 / (fan_in + fan_out))
    initial = tf.truncated_normal(shape, stddev=stddev)#正态分布
    if wd is not None:
        weight_decay = tf.multiply(tf.nn.l2_loss(initial), wd, name='weight_loss')
        tf.add_to_collection('losses', weight_decay)
    return tf.Variable(initial)


def bias_variable(shape, init_value=0.025):
    """bias_variable generates a bias variable of a given shape."""
    initial = tf.constant(init_value, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W, stride=1, padding="SAME"):
    """conv2d returns a 2d convolution layer."""
    return tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding=padding)


def complete_conv2d(input_tensor, output_channels, kernel_size, stride=1, padding="SAME", activation=tf.nn.relu, bias_init_value=0.025,
                    std_factor=1, weight_decay=None, summary=False):
    input_channels = input_tensor.get_shape().as_list()[-1]
    output_channels = int(output_channels)
    with tf.name_scope('W'):
        w_conv = weight_variable([kernel_size[0], kernel_size[1], input_channels, output_channels], std_factor=std_factor, wd=weight_decay)
        if summary:
            variable_summaries(w_conv)
    with tf.name_scope('bias'):
        b_conv = bias_variable([output_channels], init_value=bias_init_value)
        if summary:
            variable_summaries(b_conv)
    z_conv = conv2d(input_tensor, w_conv, stride=stride, padding=padding) + b_conv
    if summary:
        tf.summary.histogram('pre_activations', z_conv)
    if activation is not None:
        h_conv = activation(z_conv)
    else:
        h_conv = z_conv
    if summary:
        tf.summary.histogram('activations', h_conv)
    return h_conv

def conv_conv_pool(input_, n_filters, name="", pool=True, activation=tf.nn.elu, weight_decay=None,
                   dropout_keep_prob=None):
    """{Conv -> BN -> RELU}x2 -> {Pool, optional}

    Args:
        input_ (4-D Tensor): (batch_size, H, W, C)
        n_filters (list): number of filters [int, int]
        training (1-D Tensor): Boolean Tensor
        name (str): name postfix
        pool (bool): If True, MaxPool2D
        activation: Activation function
        weight_decay: Weight decay rate

    Returns:
        net: output of the Convolution operations
        pool (optional): output of the max pooling operations
    """
    net = input_

    with tf.variable_scope("layer_{}".format(name)):
        for i, F in enumerate(n_filters):
            net = complete_conv2d(net, F, (3, 3), padding="VALID", activation=activation,
                                           bias_init_value=-0.01,
                                           weight_decay=weight_decay,
                                           summary=False)
        if pool is False:
            return net, None
        else:
            pool = tf.layers.max_pooling2d(net, (2, 2), strides=(2, 2), name="pool_{}".format(name))
            return net, pool


def build_input_branch(input, feature_base_count, pool_count, name="", weight_decay=None):
    res_levels = pool_count + 1
    with tf.variable_scope(name):
        levels = []
        for res_level_index in range(res_levels):
            feature_count = feature_base_count * math.pow(2, res_level_index)
            if res_level_index == 0:
                # Add first level
                conv, pool = conv_conv_pool(input, [feature_count, feature_count],
                                            name="conv_pool_{}".format(res_level_index), weight_decay=weight_decay)
            elif res_level_index < res_levels - 1:
                # Add all other levels (except the last one)
                level_input = levels[-1][1]  # Select the previous pool
                conv, pool = conv_conv_pool(level_input, [feature_count, feature_count],
                                            name="conv_pool_{}".format(res_level_index), weight_decay=weight_decay)
            elif res_level_index == res_levels - 1:
                # Add last level
                level_input = levels[-1][1]  # Select the previous pool
                conv, pool = conv_conv_pool(level_input, [feature_count, feature_count],
                                            name="conv_pool_{}".format(res_level_index), pool=False,
                                            weight_decay=weight_decay)
            else:
                print("WARNING: Should be impossible to get here!")
                conv = pool = None
            levels.append((conv, pool))

    return levels

def build_common_part(branch_levels_list, feature_base_count,
                      name="", weight_decay=None):
    """
    Merges the two branches level by level in a U-Net fashion

    :param branch_levels_list:
    :param feature_base_count:
    :param name:
    :param weight_decay:
    :return:
    """
    res_levels = len(branch_levels_list[0])
    with tf.variable_scope(name):
        # Concat branches at each level + add conv layers
        levels = []
        for level_index in range(res_levels):
            concat_a_b = tf.concat([branch_levels[level_index][0] for branch_levels in branch_levels_list], axis=-1,
                                   name="concat_a_b_{}".format(level_index))
            feature_count = feature_base_count * math.pow(2, level_index)
            concat_a_b_conv, _ = conv_conv_pool(concat_a_b, [feature_count, feature_count],
                                                name="concat_a_b_conv{}".format(level_index), pool=False,
                                                weight_decay=weight_decay)
            levels.append(concat_a_b_conv)

    return levels

def build_five_param_branch(input,params,name=""):
    with tf.variable_scope(name):

        fc_input = slim.flatten(input, scope="flatten")

        # fc1
        fc1 = slim.fully_connected(fc_input, params['fc_node'], scope="fc_1")
        fc1 = slim.dropout(fc1, keep_prob=0.5)

        # fc2
        fc2 = slim.fully_connected(fc1, params['fc_node'], scope="fc_2")
        fc2 = slim.dropout(fc2, keep_prob=0.5)

        # output
        five_param_output_logits = slim.fully_connected(fc2, params['channel_output'], activation_fn=None, scope=params['name'])

        return five_param_output_logits

def transform_net(input_branch_params_list, pool_count, common_feature_base_count,output_branch_five_params_list,weight_decay=None):
    """
    Builds a multi-branch U-Net network. Has len(input_tensors) input branches and len(output_channel_counts) output branches
    """
    # Build the separate simple convolution networks for each input:
    input_branch_levels_list = []
    for params in input_branch_params_list:
        tf.summary.histogram("input_{}".format(params["name"]), params["tensor"])#tuple
        branch_levels = build_input_branch(params["tensor"], params["feature_base_count"], pool_count,
                                           name="branch_{}".format(params["name"]),
                                           weight_decay=weight_decay)
        input_branch_levels_list.append(branch_levels)

    # Build the common part of the network, concatenating inout branches at all levels
    common_part_levels = build_common_part(input_branch_levels_list,
                                           common_feature_base_count,
                                           name="common_part",
                                           weight_decay=weight_decay)


    for params in output_branch_five_params_list:
        branch_five_param_pred_output = build_five_param_branch(common_part_levels[3],params,name="branch_{}".format(params["name"]))

    return branch_five_param_pred_output


class Model:
    # TODO:loss_param?
    def __init__(self, model_name, input_res,

                 add_image_input, image_channel_count,
                 image_feature_base_count,

                 add_poly_map_input, poly_map_channel_count,
                 poly_map_feature_base_count,

                 common_feature_base_count, pool_count,

                 add_disp_output, disp_channel_count,

                 add_seg_output, seg_channel_count,

                 add_param_output,param_channel_count,

                 output_res,
                 batch_size,

                 loss_params,
                 level_loss_coefs_params,

                 learning_rate_params,
                 weight_decay,

                 image_dynamic_range, disp_map_dynamic_range_fac,
                 disp_max_abs_value):
        assert type(model_name) == str, "model_name should be a string, not a {}".format(type(model_name))
        assert type(input_res) == int, "input_res should be an int, not a {}".format(type(input_res))
        assert type(add_image_input) == bool, "add_image_input should be a bool, not a {}".format(type(add_image_input))
        assert type(image_channel_count) == int, "image_channel_count should be an int, not a {}".format(type(image_channel_count))
        assert type(image_feature_base_count) == int, "image_feature_base_count should be an int, not a {}".format(type(image_feature_base_count))
        assert type(add_poly_map_input) == bool, "add_poly_map_input should be a bool, not a {}".format(type(add_poly_map_input))
        assert type(poly_map_channel_count) == int, "poly_map_channel_count should be an int, not a {}".format(type(poly_map_channel_count))
        assert type(poly_map_feature_base_count) == int, "poly_map_feature_base_count should be an int, not a {}".format(type(poly_map_feature_base_count))
        assert type(common_feature_base_count) == int, "common_feature_base_count should be an int, not a {}".format(type(common_feature_base_count))
        assert type(pool_count) == int, "pool_count should be an int, not a {}".format(type(pool_count))
        assert type(add_disp_output) == bool, "add_disp_output should be a bool, not a {}".format(type(add_disp_output))
        assert type(disp_channel_count) == int, "disp_channel_count should be an int, not a {}".format(type(disp_channel_count))
        assert type(add_seg_output) == bool, "add_seg_output should be a bool, not a {}".format(type(add_seg_output))
        assert type(seg_channel_count) == int, "seg_channel_count should be an int, not a {}".format(type(seg_channel_count))
        assert type(add_param_output) == bool, "add_param_output should be a bool, not a {}".format(type(add_param_output))
        assert type(param_channel_count) == int, "param_channel_count should be an int, not a {}".format(type(param_channel_count))
        assert type(output_res) == int, "output_res should be an int, not a {}".format(type(output_res))
        assert type(batch_size) == int, "batch_size should be an int, not a {}".format(type(batch_size))
        assert type(loss_params) == dict, "loss_params should be a dict, not a {}".format(type(loss_params))
        assert type(level_loss_coefs_params) == list, "level_loss_coefs_params should be a list, not a {}".format(type(level_loss_coefs_params))
        assert type(learning_rate_params) == dict, "learning_rate_params should be a dict, not a {}".format(type(learning_rate_params))
        assert type(weight_decay) == float, "weight_decay should be a float, not a {}".format(type(weight_decay))
        assert type(image_dynamic_range) == list, "image_dynamic_range should be a string, not a {}".format(type(image_dynamic_range))
        assert type(disp_map_dynamic_range_fac) == float, "disp_map_dynamic_range_fac should be a float, not a {}".format(type(disp_map_dynamic_range_fac))
        assert type(disp_max_abs_value) == float or type(disp_max_abs_value) == int, "disp_max_abs_value should be a number, not a {}".format(type(disp_max_abs_value))

        # Re-init Tensorflow
        self.init_tf()
        # Init attributes from arguments
        self.model_name = model_name
        self.input_res = input_res

        self.add_image_input = add_image_input
        self.image_channel_count = image_channel_count
        self.image_feature_base_count = image_feature_base_count

        self.add_poly_map_input = add_poly_map_input
        self.poly_map_channel_count = poly_map_channel_count
        self.poly_map_feature_base_count = poly_map_feature_base_count

        self.common_feature_base_count = common_feature_base_count
        self.pool_count = pool_count

        self.add_disp_output = add_disp_output
        self.disp_channel_count = disp_channel_count

        self.add_seg_output = add_seg_output
        self.seg_channel_count = seg_channel_count

        self.add_param_output = add_param_output
        self.param_channel_count = param_channel_count

        self.output_res = output_res
        self.batch_size = batch_size
        self.weight_decay = weight_decay

        self.image_dynamic_range = image_dynamic_range
        self.disp_map_dynamic_range_fac = disp_map_dynamic_range_fac
        self.disp_max_abs_value = disp_max_abs_value

        self.input_image, \
        self.input_disp_polygon_map, \
        self.gt_seg,\
        self.rotate_theta,\
        self.scale_x,self.scale_y,\
        self.offset_x,self.offset_y = self.create_placeholders()

        input_branch_params_list = []

        input_branch_params_list.append({
            "tensor": self.input_image,
            "name": "image",
            "feature_base_count": self.image_feature_base_count,
        })
        input_branch_params_list.append({
            "tensor": self.input_disp_polygon_map,
            "name": "poly_map",
            "feature_base_count": self.poly_map_feature_base_count,
        })

        output_branch_five_params_list = []
        output_branch_five_params_list.append({
            "fc_node": 1024,
            "channel_output": self.param_channel_count,
            "name": "five_param_output",

        })

        branch_five_param_pred_output = transform_net(input_branch_params_list, self.pool_count,self.common_feature_base_count,output_branch_five_params_list,weight_decay=self.weight_decay)


        self.stacked_five_param_pred_logits = branch_five_param_pred_output

        # Create training attributes
        self.global_step = self.create_global_step()
        self.learning_rate = self.build_learning_rate(learning_rate_params)
        # Create level_coefs tensor
        # self.level_loss_coefs = self.build_level_coefs(level_loss_coefs_params)
        # Build losses
        self.total_loss = self.build_losses(loss_params)
        self.train_step = self.build_optimizer()

    @staticmethod
    def init_tf():
        tf.reset_default_graph()

    def create_placeholders(self):
        input_image = tf.placeholder(tf.float32, [self.batch_size, self.input_res, self.input_res,
                                                  self.image_channel_count])
        input_disp_polygon_map = tf.placeholder(tf.float32, [self.batch_size, self.input_res,
                                                             self.input_res,
                                                             self.poly_map_channel_count])
        gt_seg = tf.placeholder(tf.float32, [self.batch_size, self.input_res, self.input_res,
                                             self.poly_map_channel_count])
        rotate_theta = tf.placeholder(tf.float32, [self.batch_size])
        scale_x = tf.placeholder(tf.float32, [self.batch_size])
        scale_y = tf.placeholder(tf.float32, [self.batch_size])
        offset_x = tf.placeholder(tf.float32, [self.batch_size])
        offset_y = tf.placeholder(tf.float32, [self.batch_size])
        return input_image, input_disp_polygon_map, gt_seg,rotate_theta, scale_x, scale_y, offset_x, offset_y



    @staticmethod
    def create_global_step():
        return tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')

    def build_learning_rate(self, learning_rate_params):
        return tf.train.piecewise_constant(self.global_step, learning_rate_params["boundaries"],
                                           learning_rate_params["values"])

    def build_losses(self, loss_params):
        with tf.name_scope('losses'):
            l2_loss,l1_loss,warp_loss = warp_error(self.stacked_five_param_pred_logits,
                                               self.rotate_theta,self.scale_x,self.scale_y,self.offset_x,self.offset_y,
                                               self.input_disp_polygon_map,self.gt_seg)

            # tf.summary.scalar('five_param_l2_loss', l2_loss)
            tf.summary.scalar('five_param_l1_loss', l1_loss)
            tf.summary.scalar('five_param_warp_loss', warp_loss)
            # tf.add_to_collection('losses', l2_loss)
            tf.add_to_collection('losses', l1_loss)
            tf.add_to_collection('losses', warp_loss)

        total_loss = tf.add_n(tf.get_collection('losses'), name='total_loss')
        tf.summary.scalar('total_loss', total_loss)

        return total_loss

    def build_optimizer(self):
        with tf.name_scope('adam_optimizer'):
            optimizer = tf.train.AdamOptimizer(self.learning_rate)
            train_step = optimizer.minimize(self.total_loss, global_step=self.global_step)
            current_adam_lr = compute_current_adam_lr(optimizer)
            tf.summary.scalar('lr', current_adam_lr)
        return train_step


    def train(self, sess, dataset_tensors, dropout_keep_prob, with_summaries=False, merged_summaries=None,
              summaries_writer=None, summary_index=None, plot=False):

        if with_summaries:
            assert merged_summaries is not None and summaries_writer is not None, \
                "merged_summaries and writer should be specified if with_summaries is True"
        train_image, \
        _, \
        _, \
        train_gt_polygon_map, \
        train_gt_disp_field_map, \
        train_disp_polygon_map, \
        rotate_theta, \
        scale_x, scale_y, \
        offset_x, offset_y = dataset_tensors

        train_image_batch, train_gt_polygon_map_batch, train_gt_disp_field_map_batch, train_disp_polygon_map_batch,\
        rotate_theta_batch,scale_x_batch,scale_y_batch,offset_x_batch,offset_y_batch = sess.run(
            [train_image, train_gt_polygon_map, train_gt_disp_field_map, train_disp_polygon_map,rotate_theta,scale_x,scale_y,offset_x,offset_y])
        # print("gt",rotate_theta_batch,scale_x_batch,scale_y_batch,offset_x_batch,offset_y_batch)
        feed_dict = {
            self.input_image: train_image_batch,
            self.input_disp_polygon_map: train_disp_polygon_map_batch,
            self.gt_seg: train_gt_polygon_map_batch,
            self.rotate_theta: rotate_theta_batch,
            self.scale_x: scale_x_batch,
            self.scale_y: scale_y_batch,
            self.offset_x: offset_x_batch,
            self.offset_y:offset_y_batch,
        }
        # print("after",self.rotate_theta.shape)
        if with_summaries:
            if summary_index == 0:
                run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                run_metadata = tf.RunMetadata()
            else:
                run_options = run_metadata = None

            input_list = [merged_summaries, self.train_step, self.total_loss]

            if self.add_param_output:
                input_list.append(self.stacked_five_param_pred_logits)
            print("gt", rotate_theta_batch, scale_x_batch, scale_y_batch, offset_x_batch, offset_y_batch)

            output_list = sess.run(input_list, feed_dict=feed_dict, options=run_options, run_metadata=run_metadata)

            train_summary, _, train_loss = output_list[:3]

            rotate_theta_pred_batch = scale_x_pred_batch = scale_y_pred_batch = offset_x_pred_batch = offset_y_pred_batch = None
            rotate_theta_pred_batch = output_list[-1][0,0]
            scale_x_pred_batch = output_list[-1][0,1]
            scale_y_pred_batch = output_list[-1][0,2]
            offset_x_pred_batch = output_list[-1][0,3]
            offset_y_pred_batch = output_list[-1][0,4]


            summaries_writer.add_summary(train_summary, summary_index)
            if summary_index == 0:
                summaries_writer.add_run_metadata(run_metadata, 'step%03d' % summary_index)
            print("step {}, training loss = {}".format(summary_index, train_loss))
            if plot:
                train_image_batch = (train_image_batch - self.image_dynamic_range[0]) / (
                        self.image_dynamic_range[1] - self.image_dynamic_range[0])

            print("pred", rotate_theta_pred_batch, scale_x_pred_batch, scale_y_pred_batch, offset_x_pred_batch, offset_y_pred_batch)
            return train_image_batch, train_gt_polygon_map_batch, train_gt_disp_field_map_batch, train_disp_polygon_map_batch, \
                   rotate_theta_pred_batch, scale_x_pred_batch, scale_y_pred_batch, offset_x_pred_batch, offset_y_pred_batch
        else:
            _ = sess.run([self.train_step], feed_dict=feed_dict)
            return train_image_batch, train_gt_polygon_map_batch, train_gt_disp_field_map_batch, train_disp_polygon_map_batch, None, None, None, None,None, None,None, None



    def optimize(self, train_dataset_tensors, val_dataset_tensors,
                 max_iter, dropout_keep_prob,
                 logs_dir, train_summary_step, val_summary_step,
                 checkpoints_dir, checkpoint_step,
                 init_checkpoints_dirpath=None,
                 plot_results=False):
        """

        :param train_dataset_tensors:
        :param val_dataset_tensors: (If None: do not perform validation step)
        :param max_iter:
        :param dropout_keep_prob:
        :param logs_dir:
        :param train_summary_step:
        :param val_summary_step:
        :param checkpoints_dir: Directory to save checkpoints. If this is not the first time launching the optimization,
                                the weights will be restored form the last checkpoint in that directory
        :param checkpoint_step:
        :param init_checkpoints_dirpath: If this is the first time launching the optimization, the weights will be
                                     initialized with the last checkpoint in init_checkpoints_dirpath (optional)
        :param plot_results: (optional)
        :return:
        """
        # Summaries
        merged_summaries = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(os.path.join(logs_dir, "train"), tf.get_default_graph())
        val_writer = tf.summary.FileWriter(os.path.join(logs_dir, "val"), tf.get_default_graph())

        # Savers
        saver = tf.train.Saver(save_relative_paths=True)

        # The op for initializing the variables.
        init_op = tf.group(tf.global_variables_initializer(),
                           tf.local_variables_initializer())
        with tf.Session() as sess:
            sess.run(init_op)
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord)

            i = tf.train.global_step(sess, self.global_step)
            while i <= max_iter:
                if i % train_summary_step == 0:
                    time_start = time.time()

                    train_image_batch, \
                    train_gt_polygon_map_batch, \
                    train_gt_disp_field_map_batch, \
                    train_disp_polygon_map_batch, \
                    rotate_theta_pred_batch,\
                    scale_x_pred_batch,scale_y_pred_batch,\
                    offset_x_pred_batch,offset_y_pred_batch = self.train(sess, train_dataset_tensors, dropout_keep_prob,
                                                      with_summaries=True, merged_summaries=merged_summaries,
                                                      summaries_writer=train_writer, summary_index=i, plot=plot_results)
                    time_end = time.time()
                    print("\tIteration done in {}s".format(time_end - time_start))

                else:
                    self.train(sess, train_dataset_tensors, dropout_keep_prob)


                # Save checkpoint
                if i % checkpoint_step == (checkpoint_step - 1):
                    saver.save(sess, os.path.join(checkpoints_dir, self.model_name),
                               global_step=self.global_step)

                i = tf.train.global_step(sess, self.global_step)

            coord.request_stop()
            coord.join(threads)

            train_writer.close()
            val_writer.close()

    @staticmethod
    def get_output_res(input_res, pool_count):
        """
        This function has to be re-written if the model architecture changes

        :param input_res:
        :param pool_count:
        :return:
        """
        return get_output_res(input_res, pool_count)

    @staticmethod
    def get_input_res(output_res, pool_count):
        """
        This function has to be re-written if the model architecture changes

        :param output_res:
        :param pool_count:
        :return:
        """
        return get_input_res(output_res, pool_count)




def train(config,tfrecords_dirpath_list,init_run_dirpath,run_dirpath,batch_size,ds_repeat_list):

    # setup init checkpoints directory path if one is specified:
    if init_run_dirpath is not None:
        _, init_checkpoints_dirpath = run_utils.setup_run_subdirs(init_run_dirpath, config["logs_dirname"],
                                                                  config["checkpoints_dirname"])
    else:
        init_checkpoints_dirpath = None

    # setup stage run dirs
    # create run subdirectories if they do not exist
    logs_dirpath, checkpoints_dirpath = run_utils.setup_run_subdirs(run_dirpath, config["logs_dirname"],
                                                                    config["checkpoints_dirname"])

    # compute output_res
    output_res = Model.get_output_res(config["input_res"], config["pool_count"])
    print("output_res: {}".format(output_res))

    # instantiate model object (resets the default graph)
    param_model = Model(config["model_name"], config["input_res"],

                      config["add_image_input"], config["image_channel_count"],
                      config["image_feature_base_count"],

                      config["add_poly_map_input"], config["poly_map_channel_count"],
                      config["poly_map_feature_base_count"],

                      config["common_feature_base_count"], config["pool_count"],

                      config["add_disp_output"], config["disp_channel_count"],

                      config["add_seg_output"], config["seg_channel_count"],

                      config["add_param_output"],config["param_channel_count"],

                      output_res,
                      batch_size,

                      config["loss_params"],
                      config["level_loss_coefs_params"],

                      config["learning_rate_params"],
                      config["weight_decay"],

                      config["image_dynamic_range"], config["disp_map_dynamic_range_fac"],
                      config["disp_max_abs_value"])

    # train dataset
    train_dataset_filename_list = dataset_multires.create_dataset_filename_list(tfrecords_dirpath_list,
                                                                                config["tfrecord_filename_format"],
                                                                                dataset="train",
                                                                                resolution_file_repeats=ds_repeat_list)
    train_dataset_tensors = dataset_multires.read_and_decode(
        train_dataset_filename_list,
        output_res,
        config["input_res"],
        batch_size,
        config["image_dynamic_range"],
        disp_map_dynamic_range_fac=config["disp_map_dynamic_range_fac"],
        keep_poly_prob=config["keep_poly_prob"],
        data_aug=config["data_aug"],
        train=True)

    if config["perform_validation_step"]:
        # val dataset
        val_dataset_filename_list = dataset_multires.create_dataset_filename_list(tfrecords_dirpath_list,
                                                                                  config["tfrecord_filename_format"],
                                                                                  dataset="val",
                                                                                  resolution_file_repeats=ds_repeat_list)
        val_dataset_tensors = dataset_multires.read_and_decode(
            val_dataset_filename_list,
            output_res,
            config["input_res"],
            batch_size,
            config["image_dynamic_range"],
            disp_map_dynamic_range_fac=config["disp_map_dynamic_range_fac"],
            keep_poly_prob=config["keep_poly_prob"],
            data_aug=False,
            train=False)
    else:
        val_dataset_tensors = None

    # launch training
    param_model.optimize(train_dataset_tensors, val_dataset_tensors,
                             config["max_iter"], config["dropout_keep_prob"],
                             logs_dirpath, config["train_summary_step"], config["val_summary_step"],
                             checkpoints_dirpath, config["checkpoint_step"],
                             init_checkpoints_dirpath=init_checkpoints_dirpath,
                             plot_results=config["plot_results"])














def train(config,tfrecords_dirpath_list,init_run_dirpath,run_dirpath,batch_size,ds_repeat_list):

    # setup init checkpoints directory path if one is specified:
    if init_run_dirpath is not None:
        _, init_checkpoints_dirpath = run_utils.setup_run_subdirs(init_run_dirpath, config["logs_dirname"],
                                                                  config["checkpoints_dirname"])
    else:
        init_checkpoints_dirpath = None

    # setup stage run dirs
    # create run subdirectories if they do not exist
    logs_dirpath, checkpoints_dirpath = run_utils.setup_run_subdirs(run_dirpath, config["logs_dirname"],
                                                                    config["checkpoints_dirname"])

    # compute output_res
    output_res = Model.get_output_res(config["input_res"], config["pool_count"])
    print("output_res: {}".format(output_res))

    # instantiate model object (resets the default graph)
    param_model = Model(config["model_name"], config["input_res"],

                                          config["add_image_input"], config["image_channel_count"],
                                          config["image_feature_base_count"],

                                          config["add_poly_map_input"], config["poly_map_channel_count"],
                                          config["poly_map_feature_base_count"],

                                          config["common_feature_base_count"], config["pool_count"],

                                          config["add_disp_output"], config["disp_channel_count"],

                                          config["add_seg_output"], config["seg_channel_count"],

                                          config["add_param_output"],config["param_channel_count"],

                                          output_res,
                                          batch_size,

                                          config["loss_params"],
                                          config["level_loss_coefs_params"],

                                          config["learning_rate_params"],
                                          config["weight_decay"],

                                          config["image_dynamic_range"], config["disp_map_dynamic_range_fac"],
                                          config["disp_max_abs_value"])

    # train dataset
    train_dataset_filename_list = dataset_multires.create_dataset_filename_list(tfrecords_dirpath_list,
                                                                                config["tfrecord_filename_format"],
                                                                                dataset="train",
                                                                                resolution_file_repeats=ds_repeat_list)
    train_dataset_tensors = dataset_multires.read_and_decode(
        train_dataset_filename_list,
        output_res,
        config["input_res"],
        batch_size,
        config["image_dynamic_range"],
        disp_map_dynamic_range_fac=config["disp_map_dynamic_range_fac"],
        keep_poly_prob=config["keep_poly_prob"],
        data_aug=config["data_aug"],
        train=True)


    val_dataset_tensors = None

    # launch training
    param_model.optimize(train_dataset_tensors, val_dataset_tensors,
                             config["max_iter"], config["dropout_keep_prob"],
                             logs_dirpath, config["train_summary_step"], config["val_summary_step"],
                             checkpoints_dirpath, config["checkpoint_step"],
                             init_checkpoints_dirpath=init_checkpoints_dirpath,
                             plot_results=config["plot_results"])



def main(_):
    working_dir = os.path.dirname(os.path.abspath(__file__))

    # print FLAGS
    print("#--- FLAGS: ---#")
    print("config: {}".format(FLAGS.config))
    print("new_run: {}".format(FLAGS.new_run))
    print("init_run_name: {}".format(FLAGS.init_run_name))
    print("run_name: {}".format(FLAGS.run_name))
    print("batch_size: {}".format(FLAGS.batch_size))


    # load config file
    config = run_utils.load_config(FLAGS.config)

    # Check config setting coherences
    assert len(config["level_loss_coefs_params"]) == config["pool_count"], \
        "level_loss_coefs_params ({} elements) must have model_res_levels ({}) elements".format(
            len(config["level_loss_coefs_params"]), config["pool_count"])

    tfrecords_dirpath_list = [os.path.join(working_dir, tfrecords_dirpath) for tfrecords_dirpath in
                              config["tfrecords_partial_dirpath_list"]]


    ds_repeat_list = config["ds_repeat_list"]

    # setup init run directory of one is specified:
    if FLAGS.init_run_name is not None:
        init_run_dirpath = run_utils.setup_run_dir(config["runs_dirname"], FLAGS.init_run_name)
    else:
        init_run_dirpath = None

    # setup run directory:
    runs_dir = os.path.join(working_dir, config["runs_dirname"])
    current_run_dirpath = run_utils.setup_run_dir(runs_dir, FLAGS.run_name, FLAGS.new_run)

    # save config in logs directory
    run_utils.save_config(config, current_run_dirpath)

    # save FLAGS
    FLAGS_filepath = os.path.join(current_run_dirpath, "FLAGS.json")
    python_utils.save_json(FLAGS_filepath, {
        "run_name": FLAGS.run_name,
        "new_run": FLAGS.new_run,
        "batch_size": FLAGS.batch_size
    })

    train(config, tfrecords_dirpath_list, init_run_dirpath, current_run_dirpath, FLAGS.batch_size, ds_repeat_list)
if __name__ == '__main__':
    tf.app.run(main=main)