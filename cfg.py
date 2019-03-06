import os

# train_task_id = '3T256'
train_task_id = '3T736'
initial_epoch = 0
epoch_num = 24
# lr = 1e-3
lr = 5e-4
decay = 5e-4
# clipvalue = 0.5  # default 0.5, 0 means no clip
patience = 5
# load_weights = False
load_weights = True

lambda_inside_score_loss = 4.0
lambda_side_vertex_code_loss = 1.0
lambda_side_vertex_coord_loss = 1.0

total_img = 10000
validation_split_ratio = 0.1
max_train_img_size = int(train_task_id[-3:])
max_predict_img_size = int(train_task_id[-3:])  # 2400
assert max_train_img_size in [256, 384, 512, 640, 736], \
    'max_train_img_size must in [256, 384, 512, 640, 736]'
if max_train_img_size == 256:
    batch_size = 8
elif max_train_img_size == 384:
    batch_size = 4
elif max_train_img_size == 512:
    batch_size = 2
else:
    batch_size = 1
steps_per_epoch = total_img * (1 - validation_split_ratio) // batch_size
validation_steps = total_img * validation_split_ratio // batch_size

# data_dir = 'icpr/'
# origin_image_dir_name = 'image_10000/'
# origin_txt_dir_name = 'txt_10000/'
# train_image_dir_name = 'images_%s/' % train_task_id
# train_label_dir_name = 'labels_%s/' % train_task_id
# show_gt_image_dir_name = 'show_gt_images_%s/' % train_task_id
# show_act_image_dir_name = 'show_act_images_%s/' % train_task_id
# gen_origin_img = True
# draw_gt_quad = True
# draw_act_quad = True
# val_fname = 'val_%s.txt' % train_task_id
# train_fname = 'train_%s.txt' % train_task_id
# val_image_id = None

# data_dir = '/data5/xin/ocr/ICPR_text_train/train_1000'
# origin_image_dir_name = 'image_1000/'
# origin_txt_dir_name = 'txt_1000/'
# train_image_dir_name = 'images_%s/' % train_task_id
# train_label_dir_name = 'labels_%s/' % train_task_id
# show_gt_image_dir_name = 'show_gt_images_%s/' % train_task_id
# show_act_image_dir_name = 'show_act_images_%s/' % train_task_id
# gen_origin_img = True
# draw_gt_quad = True
# draw_act_quad = True
# val_fname = 'val_%s.txt' % train_task_id
# train_fname = 'train_%s.txt' % train_task_id
# val_image_id = None

data_dir = '/data5/xin/ocr/rot_boxes_v2_normal_with_class'
# same folder with different file ext
origin_image_dir_name = 'images/'
origin_txt_dir_name = 'images/'
train_image_dir_name = 'images_%s/' % train_task_id
train_label_dir_name = 'labels_%s/' % train_task_id
show_gt_image_dir_name = 'show_gt_images_%s/' % train_task_id
show_act_image_dir_name = 'show_act_images_%s/' % train_task_id
gen_origin_img = True
draw_gt_quad = True
draw_act_quad = True
val_fname = 'val_%s.txt' % train_task_id
train_fname = 'train_%s.txt' % train_task_id
val_image_id = 'IMG_1176'

# in paper it's 0.3, maybe to large to this problem
shrink_ratio = 0.2
# pixels between 0.2 and 0.6 are side pixels
shrink_side_ratio = 0.6
epsilon = 1e-4

num_channels = 3
feature_layers_range = range(5, 1, -1)
# feature_layers_range = range(3, 0, -1)
feature_layers_num = len(feature_layers_range)
# pixel_size = 4
pixel_size = 2 ** feature_layers_range[-1]
locked_layers = False


# model_weights_path = 'model/weights_%s.{epoch:03d}-{val_loss:.3f}.h5' \
#                      % train_task_id
# saved_model_file_path = 'saved_model/east_model_%s.h5' % train_task_id
# saved_model_weights_file_path = 'saved_model/east_model_weights_%s.h5'\
#                                 % train_task_id

root_dir = '/data5/xin/ocr/train/vgg_avd_0305'
model_dir = os.path.join(root_dir, 'model')
saved_model_dir = os.path.join(root_dir, 'saved_model')

for d in [root_dir, model_dir, saved_model_dir]:
	if not os.path.exists(d):
	    os.mkdir(d)

model_weights_path = os.path.join(model_dir, 'weights_%s.{epoch:03d}-{val_loss:.3f}.h5' \
                     % train_task_id)
saved_model_file_path = os.path.join(saved_model_dir, 'east_model_%s.h5' % train_task_id)
saved_model_weights_file_path = os.path.join(saved_model_dir, 'east_model_weights_%s.h5'\
                                % train_task_id)

pixel_threshold = 0.9
side_vertex_pixel_threshold = 0.9
trunc_threshold = 0.1
predict_cut_text_line = False
predict_write2txt = True
