import tensorflow as tf
import numpy as np
import pickle
import time
import argparse
import os
from model import DPN

def parse_args():

	parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	parser.add_argument('--inner-horizon', type=int, default=5, help='length of RNN rollout horizon')
	parser.add_argument('--outer-horizon', type=int, default=5, help='length of BC loss horizon')
	parser.add_argument('--sampling-max-horizon', type=int, default=0, help='max length of BC loss horizon for sampling')
	parser.add_argument('--num-plan-updates', type=int, default=8, help='number of planning update steps before BC loss')
	parser.add_argument('--n-hidden', type=int, default=1, help='number of hidden layers to encode after conv')
	parser.add_argument('--obs-latent-dim', type=int, default=128, help='obs latent space dim')
	parser.add_argument('--act-latent-dim', type=int, default=128, help='act latent space dim')
	parser.add_argument('--meta-gradient-clip-value', type=float, default=25., help='meta gradient clip value')
	parser.add_argument('--batch-size', type=int, default=128, help='batch size')
	parser.add_argument('--test-batch-size', type=int, default=128, help='test batch size')
	parser.add_argument('--il-lr-0', type=float, default=0.5, help='il_lr_0')
	parser.add_argument('--il-lr', type=float, default=0.25, help='il_lr')
	parser.add_argument('--ol-lr', type=float, default=0.0035, help='ol_lr')
	parser.add_argument('--num-batch-updates', type=int, default=100000, help='number of minibatch updates')
	parser.add_argument('--testing-frequency', type=int, default=500, help='how frequently to get stats for test data')
	parser.add_argument('--log-frequency', type=int, default=1000, help='how frequently to log the data')
	parser.add_argument('--log-file', type=str, default='log', help='name of log file to dump test data stats')
	parser.add_argument('--log-directory', type=str, default='/scr/kevin/unsupervised_upn/', help='name of log directory to dump checkpoints')
	parser.add_argument('--huber', dest='huber_loss', action='store_true', help='whether to use Huber Loss')
	parser.add_argument('--no-huber', dest='huber_loss', action='store_false', help='whether not to use Huber Loss')
	parser.set_defaults(huber_loss=True)
	parser.add_argument('--huber-delta', type=float, default=1., help='delta coefficient in Huber Loss')
	parser.add_argument('--img-c', type=int, default=3, help='number of channels in input')
	parser.add_argument('--task', type=str, default='pointmass', help='which task to train on')
	parser.add_argument('--act-scale-coeff', type=float, default=1., help='scaling factor for actions')
	parser.add_argument('--act-dim', type=int, default=2, help='dimensionality of action space')
	parser.add_argument('--joint-dim', type=int, default=4, help='dimensionality of joint space')
	parser.add_argument('--img-h', type=int, default=84, help='image height')
	parser.add_argument('--img-w', type=int, default=84, help='image width')
	parser.add_argument('--num-train', type=int, default=5000, help='number of rollouts for training')
	parser.add_argument('--num-test', type=int, default=1000, help='number of rollouts for test')
	parser.add_argument('--spatial-softmax', dest='spatial_softmax', action='store_true', help='whether to use spatial softmax')
	parser.add_argument('--bias-transform', dest='bias_transform', action='store_true', help='whether to use bias transform')
	parser.add_argument('--bt-num-units', type=int, default=10, help='number of dimensions in bias transform')
	parser.add_argument('--nonlinearity', type=str, default='swish', help='which nonlinearity for dynamics and fully connected')
	parser.add_argument('--decay', type=str, default='None', help='decay the learning rate in the outer loop')
	parser.add_argument('--niter-init', type=int, default=50000, help='number of iterations of running the initial learning rate')
	parser.add_argument('--learn-lr', type=str, default='False', help='learn the il-lr')
	parser.add_argument('--test', type=str, default='False', help='restore the model for testing or not')
	parser.add_argument('--dt', type=int, default=1, help='number of time steps between the initial image and the final goal image')
	parser.add_argument('--restore-iter', type=int, default=-1, help='restore the checkpoint at which iteration')
	parser.add_argument('--date', type=str, default='False', help='date of the checkpoint created')
	parser.add_argument('--training-tfrecord', type=str, default='None', help='path to the tfrecord file for training')
	parser.add_argument('--validation-tfrecord', type=str, default='None', help='path to the tfrecord file for validation')
	parser.add_argument('--n-hidden-act', type=int, default=1, help='number of hidden layers for action')
	parser.add_argument('--beta', type=float, default=1.0, help='beta for beta-vae')
	args = parser.parse_args()
	return args

def decode_and_sample(serialized_example,
				 img_h=100,
				 img_w=100,
				 act_dim=2,
				 joint_dim=4,
				 max_horizon=24,
				 sampling_max_horizon=None,
				 img_c=3,
				 act_scale_coeff=1.,
				 dt=1,
				 task='reacher'):
	"""Parses an image and label from the given `serialized_example`."""
	if sampling_max_horizon == 0:
		sampling_max_horizon = max_horizon
	features = tf.parse_single_example(
		serialized_example,
		# Defaults are not specified since both keys are required.
		features={
			'images': tf.FixedLenFeature([], tf.string),
			'actions': tf.FixedLenFeature([sampling_max_horizon*act_dim], tf.float32) if 'fetch' not in task else tf.FixedLenFeature([sampling_max_horizon*(act_dim+1)], tf.float32),
			'qts': tf.FixedLenFeature([sampling_max_horizon*joint_dim], tf.float32),
			})

	# Decode and normalize images.
	images = tf.decode_raw(features['images'], tf.uint8)
	images.set_shape(((sampling_max_horizon+1) * img_h * img_w * img_c,))
	images = tf.reshape(images, [sampling_max_horizon+1, img_h, img_w, img_c])
	images = tf.cast(images, tf.float32) / 255.0

	# Decode actions and states.
	actions = tf.cast(features['actions'], tf.float32)
	if 'fetch' not in task:
		actions = tf.reshape(actions, [sampling_max_horizon, act_dim])
	else:
		actions = tf.reshape(actions, [sampling_max_horizon, (act_dim+1)])
		actions = actions[:, :-1] # z velocity is always 0
	qt = tf.cast(features['qts'], tf.float32)
	qt = tf.reshape(qt, [sampling_max_horizon, joint_dim])

	# Obtain the initial image and the goal image
	t1 = tf.random_uniform(shape=[], minval=0, maxval=sampling_max_horizon+1-dt, dtype=tf.int32)
	t2 = t1 + dt
	effective_horizon = t2-t1
	ot = images[t1]
	og = images[t2]

	# Obtain the action and state sequences
	atT = tf.concat(values=[(1./act_scale_coeff)*actions[t1:t2], tf.zeros((max_horizon-effective_horizon, act_dim))], axis=0)
	qtT = qt[t1]
	mask = tf.concat(values=[tf.ones((effective_horizon)), tf.zeros((max_horizon-effective_horizon))], axis=0) + tf.zeros((max_horizon))
	eff_horizon = effective_horizon - 1
	return ot, og, qtT, atT, mask, eff_horizon

def main():
	args = parse_args()
	test = True if args.test == 'True' else False
	decay_str = '' if args.decay == 'None' else args.decay
	learn_lr_str = '' if args.learn_lr == 'False' else 'learn_lr'
	log_directory = args.log_directory #YOUR TRAINING DATA PATH GOES HERE #
	act_scale_coeff = args.act_scale_coeff
	dirname = args.task + '_latent_planning_ol_lr' + str(args.ol_lr) + '_il_lr' + str(args.il_lr) + '_num_plan_updates_' + \
			str(args.num_plan_updates) + '_horizon_' + str(args.inner_horizon) + '_num_train_' + str(args.num_train) + '_'  + \
			decay_str + '_' + learn_lr_str + '_clip' + str(args.meta_gradient_clip_value) + \
			'_n_hidden_' + str(args.n_hidden) + '_latent_dim_' + str(args.obs_latent_dim) + '_dt_' + str(args.dt)
	if args.spatial_softmax:
		dirname += '_fp'
	if args.bias_transform:
		dirname += '_bt'
	dirname += '_n_act_%d' % args.n_hidden_act + '_act_latent_dim_' + str(args.act_latent_dim)
	dirname += '_beta_' + str(args.beta)
	if not test:
		dirname +=  '_' + time.strftime("%d-%m-%Y_%H-%M-%S")
	else:
		dirname += '_' + args.date
	if not os.path.isdir(log_directory+'summ/' + dirname) and not test:
		os.makedirs(log_directory+'summ/' + dirname)
	tf_config = tf.ConfigProto()
	tf_config.gpu_options.allow_growth=True
	sess = tf.Session(config=tf_config)

	# Default hyperparameters for the convolutional neural network
	conv_params = [(64,5,2,'SAME'),(64,5,2,'SAME'),(64,5,1,'SAME'),(64,5,1,'SAME')]

	if not test:
		import glob
		assert args.training_tfrecord != 'None'
		filenames = [glob.glob(args.training_tfrecord)] +[glob.glob(args.validation_tfrecord)]
		iterators = []
		for filename in filenames:
			dataset = tf.data.TFRecordDataset(filename)
			if 'train' in filename:
				iter_per_epoch = args.num_train // args.batch_size
			else:
				iter_per_epoch = args.num_test // args.batch_size
			dataset = dataset.repeat(count=args.num_batch_updates // (iter_per_epoch) + 2)
			dataset = dataset.shuffle(500+3*args.batch_size)
			dataset = dataset.map(lambda example: decode_and_sample(example,
																	img_h=args.img_h,
																	img_w=args.img_w,
																	act_dim=args.act_dim,
																	joint_dim=args.joint_dim,
																	max_horizon=args.inner_horizon,
																	sampling_max_horizon=args.sampling_max_horizon,
																	act_scale_coeff=act_scale_coeff,
																	img_c=args.img_c,
																	dt=args.dt,
																	task=args.task),
																	num_parallel_calls=10)
			dataset = dataset.batch(args.batch_size)
			print('Processed dataset')
			iterators.append(dataset.make_one_shot_iterator())
		train_iterator, test_iterator = iterators[0], iterators[1]

		imp_networks = [DPN(img_w=args.img_w,
						  img_h=args.img_h,
						  img_c=args.img_c,
						  act_dim=args.act_dim,
						  joint_dim=args.joint_dim,
						  inner_horizon=args.inner_horizon,
						  outer_horizon=args.outer_horizon,
						  num_plan_updates=args.num_plan_updates,
						  conv_params=conv_params,
						  n_hidden=args.n_hidden,
						  n_hidden_act=args.n_hidden_act,
						  obs_latent_dim=args.obs_latent_dim,
						  act_latent_dim=args.act_latent_dim,
						  if_huber=args.huber_loss,
						  delta_huber=args.huber_delta,
						  meta_gradient_clip_value=args.meta_gradient_clip_value,
						  spatial_softmax=args.spatial_softmax,
						  bias_transform=args.bias_transform,
						  bt_num_units=args.bt_num_units,
						  decay=args.decay,
						  niter_init=args.niter_init,
						  niter_decay=args.num_batch_updates - args.niter_init,
						  learn_lr=True if args.learn_lr == 'True' else False,
						  init_il_lr=args.il_lr,
						  beta=args.beta,
						  use_tfrecord=True,
						  train=True if i == 0 else False,
						  iterator=iterators[i],
						  test=False) for i in range(len(iterators))]
	else:
		imp_network = DPN(img_w=args.img_w,
						  img_h=args.img_h,
						  img_c=args.img_c,
						  act_dim=args.act_dim,
						  joint_dim=args.joint_dim,
						  inner_horizon=args.inner_horizon,
						  outer_horizon=args.outer_horizon,
						  num_plan_updates=args.num_plan_updates,
						  conv_params=conv_params,
						  n_hidden=args.n_hidden,
						  n_hidden_act=args.n_hidden_act,
						  obs_latent_dim=args.obs_latent_dim,
						  act_latent_dim=args.act_latent_dim,
						  if_huber=args.huber_loss,
						  delta_huber=args.huber_delta,
						  meta_gradient_clip_value=args.meta_gradient_clip_value,
						  spatial_softmax=args.spatial_softmax,
						  bias_transform=args.bias_transform,
						  bt_num_units=args.bt_num_units,
						  decay=args.decay,
						  niter_init=args.niter_init,
						  niter_decay=args.num_batch_updates - args.niter_init,
						  learn_lr=True if args.learn_lr == 'True' else False,
						  init_il_lr=args.il_lr,
						  beta=args.beta,
						  test=True)

	sess.run(tf.global_variables_initializer())

	saver = tf.train.Saver(max_to_keep=40)
	if test:
		try:
			model_file = tf.train.latest_checkpoint(log_directory+'summ/' + dirname + '/models/')
		except:
			import glob
			assert args.restore_iter > 0
			model_file = log_directory+'summ/' + dirname + '/models/model_%d' % args.restore_iter
		if args.restore_iter > 0:
			model_file = model_file[:model_file.index('model_')] + 'model_' + str(args.restore_iter)
		if model_file:
			ind1 = model_file.index('model_')
			resume_itr = int(model_file[ind1+6:])
			print("Restoring model weights from " + model_file)
			saver.restore(sess, model_file)
			saver.save(sess, model_file.replace('model_', 'model_plan_test_'))
	else:
		train_writer = tf.summary.FileWriter(log_directory+'summ/' + dirname, sess.graph)
		for batch_idx  in range(args.num_batch_updates):
			imp_networks[0].train(args.il_lr_0, 
							  args.il_lr,
							  args.ol_lr,
							  sess)

			if batch_idx % args.testing_frequency == 0:
				vae_loss_train, kl_div_loss_train, bc_loss_train, plan_loss_train, latent_xpred_t, latent_xg_t, bc_loss_first_step_train, train_summ = imp_networks[0].stats(args.il_lr_0,
																																											  args.il_lr,
																																											  sess)

				vae_loss, kl_div_loss, bc_loss, plan_loss, latent_xpred, latent_xg, bc_loss_first_step, val_summ = imp_networks[1].stats(args.il_lr_0,
																																		args.il_lr,
																																		sess)
				train_writer.add_summary(train_summ, batch_idx)
				train_writer.add_summary(val_summ, batch_idx)
				print("Batch Update", batch_idx,
					  "VAE Loss", vae_loss,
					  "kl_div Loss", kl_div_loss,
					  "BC Loss", bc_loss,
					  "BC Loss First Step", bc_loss_first_step,
					  "VAE Loss train", vae_loss_train,
					  "kl_div Loss train", kl_div_loss_train,
					  "logprob Loss train", logprob_loss_train,
					  "BC Loss Train", bc_loss_train,
					  "BC Loss First Step Train", bc_loss_first_step_train,
					  "Plan Loss", plan_loss)

				if batch_idx % args.log_frequency == 0:
					save_path = saver.save(sess, log_directory+'summ/' + dirname + '/models/model_%d' % batch_idx)
					print("Model saved in path: %s" % save_path)

if __name__ == '__main__':
	main()
