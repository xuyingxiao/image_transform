# image_transform
This is a demo.It exists because that tf.contrib.image.transform cannot back propagation gradients into transformation parameters.
While I want to pred and updated this parameters by deep learning and I think is derivable if interpolation is ignored.
Input:
	input_image[batch_size, 220,220,3]
	input_disp_polygon_map[batch_size, 220,220,3]
	gt_seg [batch_size, 220,220,3]
	rotate_theta [batch_size, 1]
	scale_x  [batch_size, 1]
	scale_y  [batch_size, 1]
	offset_x [batch_size, 1]
	offset_y  [batch_size, 1]
Output:
	branch_five_param_pred_output is the five parameters mentioned.[batch_size,5]
Network:
This demo uses a simple network. 2 branches input (input image & input_disp_polygon_map) conv and then concat, and then 2 fc layers (500 nodes),
and output a [batch_size,5] tensor with pred_rotate_theta,pred_scale_x,pred_scale_x,pred_offset_x,pred_offset_y
Loss:
1.l1_loss
    l1_loss = tf.reduce_mean(tf.abs(pred_rotate_theta - gt_rotate_theta) + tf.abs(pred_scale_x - gt_scale_x) + tf.abs(pred_scale_y - gt_scale_y) + tf.abs(pred_offset_x - gt_offset_x) + tf.abs(pred_offset_y - gt_offset_y))
2.warp_loss
I conducted the homo matrix by pred parameters and applied to the input_disp_polygon_map,aimed to make these two images/tensors(warp image & seg_gt) as similar as possible.
While it can't run if there is only warp_loss because "No gradients provided for any variable", and runs if there are l1_loss+warp_loss with warp_loss not converge.

    h10 = tf.cos(pred_rotate_theta)*pred_scale_x
    h11 = tf.sin(pred_rotate_theta)*pred_scale_x
    h12 = pred_offset_x
    h20 = -tf.sin(pred_rotate_theta)*pred_scale_x
    h21 = tf.cos(pred_rotate_theta)*pred_scale_y
    h22 = pred_offset_y
    h30 = tf.constant([0.0])
    h31 = tf.constant([0.0])
    h32 = tf.constant([1.0])

    matrix_3_3 = tf.linalg.inv(tf.reshape(tf.concat([h10,h11,h12,h20,h21,h22,h30,h31,h32],axis=-1),shape=(3,3)))
    matrix_1_8 = tf.contrib.image.matrices_to_flat_transforms(matrix_3_3)

    warp_image = tf.contrib.image.transform(input_disp_polygon_map,matrix_1_8,interpolation="BILINEAR",name=None)
    print("warp_image",warp_image.shape)
	warp_loss = tf.reduce_mean(tf.abs(warp_image - input_gt_polygon_map))
	
In order to test,the batch_size=1,and increase the batch_size will make the homo matrix wrong dimension.