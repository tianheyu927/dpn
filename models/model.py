import tensorflow as tf
import numpy as np
import tensorflow.contrib.layers as layers
from tf_utils import log_normal, log_bernoulli
tfd = tf.contrib.distributions


class DPN(object):

	def __init__(self,
			img_w=64,
			img_h=64,
			img_c=6,
			act_dim=2,
			joint_dim=4,
			inner_horizon=5,
			outer_horizon=2,
			num_plan_updates=5,
			conv_params=[(40,3,2,'VALID'), (40,3,2,'VALID'), (40, 3, 2, 'VALID')],
			n_hidden=2,
			n_hidden_act=2,
			n_hidden_dynamics=1,
			obs_latent_dim=128,
			act_latent_dim=64,
			meta_gradient_clip_value=10,
			if_huber=True,
			delta_huber=1.,
			bt_num_units=10,
			bias_transform=False,
			nonlinearity='swish',
			spatial_softmax=False,
			decay='None',
			niter_init=0,
			niter_decay=100000,
			learn_lr=False,
			init_il_lr=0.25,
			beta=0.5,
			use_tfrecord=False,
			train=True,
			iterator=None,
			test=False):

		######## INITIALIZE THE INPUTS #########

		# Input Parameters
		self.img_w = img_w
		self.img_h = img_h
		self.act_dim = act_dim
		self.inner_horizon = inner_horizon
		self.outer_horizon = outer_horizon
		self.num_plan_updates = num_plan_updates
		self.learn_lr = learn_lr

		self.ol_lr = tf.placeholder(tf.float32, name = 'ol_lr')

		if iterator is not None:
			self.ot, self.og, self.qt, self.atT_target, self.plan_loss_mask, self.eff_horizons = iterator.get_next()
		else:
			#Input Tensor Placeholders
			self.ot = tf.placeholder(tf.float32, [None, img_h, img_w, img_c], name = 'ot')
			self.og = tf.placeholder(tf.float32, [None, img_h, img_w, img_c], name = 'og')
			self.atT_target = tf.placeholder(tf.float32, [None, inner_horizon, act_dim], name='atT_target')
			#need joint info?
			self.qt = tf.placeholder(tf.float32, [None, joint_dim], name='qt')
			self.plan_loss_mask = tf.placeholder(tf.float32, [None, inner_horizon], name='mask')
			self.eff_horizons = tf.placeholder(tf.int32, [None], name='eff_horizons')
		#Copy placeholder for repeated plan updates
		self.atT = None

		#######  BUILD THE COMPUTATIONAL GRAPH #######

		print('Encoding observations and goals')
		#Part 1: Encode the observation and the goal in a latent space, in two stages - first in conv layer and then in fully connected layer
		with tf.variable_scope('gradplanner') as training_scope:
				if train is False:
					training_scope.reuse_variables()
				#inner learning rate
				if not learn_lr:
					self.il_lr_0 = tf.placeholder(tf.float32, name = 'il_lr_0')
					self.il_lr = tf.placeholder(tf.float32, name = 'il_lr')
				else:
					self.il_lr = [tf.maximum(tf.get_variable('il_lr_%d' % j,
											shape=[],
											initializer=tf.constant_initializer(init_il_lr),
											dtype=tf.float32), 0) for j in range(self.num_plan_updates)]
					self.il_lr_0 = self.il_lr[0]
					self.il_lr = self.il_lr[1:]
				xt = self._encode_conv(self.ot,
									   conv_params,
									   scope='obs_conv_encoding',
									   layer_norm=True,
									   nonlinearity='swish',
									   spatial_softmax=spatial_softmax,
									   reuse=False if train else True)
				xt = self._encode_fc(xt,
									 n_hidden=n_hidden,
									 scope='obs_fc_encoding',
									 layer_norm=True,
									 latent_dim=obs_latent_dim,
									 nonlinearity=nonlinearity,
									 #nonlinearity='swish',
									 reuse=False if train else True)

				xg = self._encode_conv(self.og,
									 conv_params,
									 scope='obs_conv_encoding',
									 layer_norm=True,
									 nonlinearity='swish',
									 spatial_softmax=spatial_softmax,
									 reuse=True)
				xg = self._encode_fc(xg,
									 n_hidden=n_hidden,
									 scope='obs_fc_encoding',
									 layer_norm=True,
									 latent_dim=obs_latent_dim,
									 nonlinearity=nonlinearity,
									 reuse=True)

				# concat visual features with state information
				xt = tf.concat([xt, self.qt], axis=1)

				if bias_transform:
							bias_transform = tf.get_variable('bias_transform',
															 [1,bt_num_units],
															 initializer=tf.constant_initializer(0.1))
							bias_transform = tf.tile(bias_transform, multiples=tf.stack([tf.shape(xt)[0], 1]))
							xt = tf.concat([xt, bias_transform], 1)

				xt = self._fully_connected(xt,
										   n_hidden=n_hidden,
										   scope='joint_encoding',
										   out_dim=obs_latent_dim,
										   nonlinearity=nonlinearity,
										   layer_norm=True,
										   reuse=False if train else True)

		#Part 2: Encode the action in the latent space via VAE.
				self.utT, logqzx, logpz = self._sample_and_log_prob_zx(self.atT_target,
														scope='plan_encoding',
														n_hidden=n_hidden_act,
														latent_dim=act_latent_dim,
														horizon=inner_horizon,
														act_dim=act_dim,
														nonlinearity=nonlinearity,
														reuse=False if train else True)

		

				print('Rolling out the latent plan')
		#Part 3: Rollout the latent plan over the planning horizon
				self._rollout_plan_in_latent_space( xt,
													xg,
													self.eff_horizons,
													self.il_lr_0,
													self.il_lr,
													num_plan_updates=num_plan_updates,
													horizon=inner_horizon,
													n_hidden_dynamics=n_hidden_dynamics,
													scope='rollout',
													act_dim=act_dim,
													obs_latent_dim=obs_latent_dim,
													act_latent_dim=act_latent_dim,
													nonlinearity=nonlinearity,
													if_huber=if_huber,
													delta_huber=delta_huber,
													meta_gradient_clip_value=meta_gradient_clip_value,
													layer_norm=True,
													reuse=False if train else True)

				self.atT, logpxz = self._sample_and_log_prob_xz(self.utT,
																self.atT_target,
																scope='plan_decoding',
																n_hidden=n_hidden_act,
																latent_dim=act_latent_dim,
																horizon=inner_horizon,
																act_dim=act_dim,
																nonlinearity=nonlinearity,
																reuse=False if train else True)

		#Part 4: Compute the DPN loss
		if train:
			prefix = 'Training '
		else:
			prefix = 'Validation '
		print('Computing losses')
		error = tf.reduce_sum(tf.square(self.atT - self.atT_target), reduction_indices=[2])
		error = error * self.plan_loss_mask
		bc_loss = tf.reduce_sum(error[:, :outer_horizon], reduction_indices=[1]) / tf.reduce_sum(self.plan_loss_mask, reduction_indices=[1])
		bc_loss = tf.reduce_mean(bc_loss)
		bc_loss_one_step = tf.reduce_mean(error[:, 0])

		encoder_loss = logqzx - logpz
		decoder_loss = -logpxz
		kl_div = tf.reduce_mean(encoder_loss)
		loss = bc_loss + beta*kl_div

		#Part 5: Training and Diagnostics Ops
		if train:
			global_step = tf.Variable(0, trainable=False)
			if decay != 'None':
				if decay == 'linear':
					post_burnin_learning_rate = tf.train.polynomial_decay(self.ol_lr, global_step - niter_init,
																		  niter_decay, 0.0,
																		  power=1.0)
					ol_lr = tf.maximum(tf.where(
					  tf.less(tf.cast(global_step, tf.int32), tf.constant(niter_init)),
					  self.ol_lr,
					  post_burnin_learning_rate), 0.0, name='learning_rate')
				else:
					ol_lr = tf.train.noisy_linear_cosine_decay(
					  self.ol_lr, global_step, niter_decay)
			else:
				ol_lr = self.ol_lr
			optimizer = tf.train.AdamOptimizer(ol_lr)
			encoder_vars = [var for var in tf.global_variables() if 'plan_encoding' in var.name]
			decoder_vars = [var for var in tf.global_variables() if 'plan_encoding' not in var.name]
			encoder_grads_and_vars = optimizer.compute_gradients(loss, encoder_vars)
			decoder_grads_and_vars = optimizer.compute_gradients(loss, decoder_vars)

			self.train_op = tf.group(optimizer.apply_gradients(encoder_grads_and_vars),
									optimizer.apply_gradients(decoder_grads_and_vars))
		self.get_inner_loss_op = self.plan_loss
		self.get_outer_loss_op = bc_loss
		self.get_outer_loss_first_step_op = bc_loss_one_step
		self.get_vae_loss_op = loss
		self.get_kl_div_loss_op = kl_div
		self.get_plan_op = self.atT
		plan_op = tf.identity(self.get_plan_op, name='plan')
		self.get_xt = xt
		self.get_xg = xg
		if test:
			xt_dummy = tf.identity(xt, name='xt')
			xg_dummy = tf.identity(xg, name='xg')

		if iterator is not None:
			prefix = 'Training ' if train else 'Validation '
			summ = [tf.summary.scalar(prefix + 'vae loss', loss), tf.summary.scalar(prefix + 'kl_div loss', kl_div),
					tf.summary.scalar(prefix + 'BC loss', tf.sqrt(bc_loss)), tf.summary.scalar(prefix + 'plan loss', tf.sqrt(self.plan_loss)),
					tf.summary.scalar(prefix + 'BC loss first step', tf.sqrt(bc_loss_one_step)), tf.summary.image(prefix + 'initial image', self.ot, max_outputs=5),
					tf.summary.image(prefix + 'goal image', self.og, max_outputs=5)]
			if 'Training' in prefix:
				for k in range(num_plan_updates):
					summ.append(tf.summary.histogram('Gradient_step_%d' % k, self.atT_grads[k]))
			self.summ_op = tf.summary.merge(summ)
		print('Done building the graph')

	@property
	def trainable_vars(self):
		weights = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="gradplanner")
		return weights

	def _rollout_plan_in_latent_space(self,
										xt,
										xg,
										eff_horizons,
										il_lr_0,
										il_lr,
										num_plan_updates=5,
										horizon=5,
										scope='rollout',
										n_hidden_dynamics=1,
										obs_latent_dim=16,
										act_latent_dim=16,
										act_dim=2,
										nonlinearity='swish',
										if_huber=True,
										delta_huber=1.,
										meta_gradient_clip_value=10,
										encode_action=True,
										layer_norm=True,
										reuse=False):

		with tf.variable_scope(scope, reuse=reuse):
			self.atT_grads = []
			for update_idx in range(num_plan_updates):
				xg_pred = xt
				xg_preds = []
				if update_idx == 0:
					plan_encode_scope_reuse = False
				else:
					plan_encode_scope_reuse = True

				for time_idx in range(0, horizon):
					if time_idx >= 1 or update_idx >=1:
						dynamics_scope_reuse = True
					else:
						dynamics_scope_reuse = False

					xg_pred = self._fully_connected(tf.concat([xg_pred, self.utT[:, time_idx, :]], axis=1),
													n_hidden=n_hidden_dynamics,
													out_dim=obs_latent_dim,
													scope='dynamics',
													nonlinearity=nonlinearity,
													reuse=dynamics_scope_reuse)

					xg_preds.append(xg_pred)
				xg_preds = tf.convert_to_tensor(xg_preds) # horizon * batch_size*obs_latent_dim
				xg_preds = tf.transpose(xg_preds, [1,0,2]) # batch_size * horizon * obs_latent_dim
				xg_pred = tf.gather_nd(xg_preds, tf.concat([tf.expand_dims(tf.range(tf.shape(xg_preds)[0]),1), tf.expand_dims(eff_horizons, 1)],1))
				if if_huber:
					self.plan_loss = tf.reduce_sum(tf.losses.huber_loss(xg, xg_pred, delta=delta_huber, reduction="none"), reduction_indices=[1]) # Trying Huber Loss
				else:
					self.plan_loss = tf.reduce_sum(tf.square(xg_pred - xg), reduction_indices=[1])
				self.plan_loss = tf.reduce_mean(self.plan_loss)
				atT_grad = tf.gradients(self.plan_loss, self.utT)[0]
				atT_grad = tf.clip_by_value(atT_grad, -meta_gradient_clip_value, meta_gradient_clip_value)
				self.atT_grads.append(atT_grad)
				if update_idx == 0:
					self.utT = self.utT - il_lr_0*atT_grad
				else:
					if self.learn_lr:
						self.utT = self.utT - il_lr[update_idx-1]*atT_grad
					else:
						self.utT = self.utT - il_lr*atT_grad
			self.xg_pred = xg_pred

	def _encode_conv(self,
						x,
						conv_params,
						scope='obs_conv_encoding',
						layer_norm=False,
						nonlinearity='swish',
						spatial_softmax=False,
						reuse=False):

		with tf.variable_scope(scope, reuse=reuse):
			out = x
			for num_outputs, kernel_size, stride, padding in conv_params:
				out = layers.convolution2d( out,
											num_outputs=num_outputs,
											kernel_size=kernel_size,
											stride=stride,
											padding=padding,
											activation_fn=None)

				if layer_norm is True:
					out = layers.layer_norm(out)

				# Apply the non-linearity after layer-norm
				if nonlinearity == 'swish':
					out = tf.nn.sigmoid(out)*out #swish non-linearity
				elif nonlinearity == 'relu':
					out = tf.nn.relu(out)
			if spatial_softmax:
				shape = tf.shape(out)
				static_shape = out.shape
				height, width, num_channels = shape[1], shape[2], static_shape[3]
				pos_x, pos_y = tf.meshgrid(tf.linspace(-1., 1., num=height),
											   tf.linspace(-1., 1., num=width),
										indexing='ij')
				pos_x = tf.reshape(pos_x, [height*width])
				pos_y = tf.reshape(pos_y, [height*width])
				out = tf.reshape(tf.transpose(out, [0,3,1,2]), [-1, height*width])
				softmax_attention = tf.nn.softmax(out)
				expected_x = tf.reduce_sum(pos_x*softmax_attention, [1], keep_dims=True)
				expected_y = tf.reduce_sum(pos_y*softmax_attention, [1], keep_dims=True)
				expected_xy = tf.concat([expected_x, expected_y], 1)
				feature_keypoints = tf.reshape(expected_xy, [-1, num_channels.value*2])
				feature_keypoints.set_shape([None, num_channels.value*2])
				return feature_keypoints
			else:
				out = layers.flatten(out) # flatten the conv output
				return out

	def _encode_fc(self,
					x,
					scope='obs_fc_encoding',
					n_hidden=2,
					layer_norm=True,
					latent_dim=16,
					nonlinearity='swish',
					reuse=False):

		with tf.variable_scope(scope, reuse=reuse):
			out = x
			for _ in range(n_hidden):
				out = layers.fully_connected(   out,
												num_outputs=latent_dim,
												activation_fn=None)
				if layer_norm is True:
					out = layers.layer_norm(out, center=True, scale=True)
				if nonlinearity == 'swish':
					out = tf.nn.sigmoid(out)*out
				elif nonlinearity == 'relu':
					out = tf.nn.relu(out)

			return out

	def _encode_plan(self,
					plan,
					scope='plan_encoding',
					n_hidden=1,
					nonlinearity='swish',
					latent_dim=16,
					horizon=5,
					act_dim=2,
					layer_norm=True,
					reuse=False):

		# encode it into a plan
		with tf.variable_scope(scope, reuse=reuse):
			out = plan
			out = tf.reshape(out, [-1, act_dim])
			for _ in range(n_hidden):
				if _ != n_hidden - 1:
					out = layers.fully_connected(out,
													num_outputs=latent_dim,
													activation_fn=None)
					if layer_norm is True:
						out = layers.layer_norm(out, center=True, scale=True)
					if nonlinearity == 'swish':
						out = tf.nn.sigmoid(out)*out
					if nonlinearity == 'relu':
						out = tf.nn.relu(out)
					if nonlinearity == 'tanh':
						out = tf.nn.tanh(out)
				else:
					mean = layers.fully_connected(out,
													num_outputs=latent_dim,
													activation_fn=None)
					logstd = layers.fully_connected(out,
													num_outputs=latent_dim,
													activation_fn=None)
			mean = tf.reshape(mean, [-1, horizon, latent_dim])
			std = tf.exp(tf.reshape(logstd, [-1, horizon, latent_dim]))
			return mean, std

	def _decode_plan(self,
					utT,
					scope='plan_decoding',
					n_hidden=1,
					nonlinearity='swish',
					latent_dim=16,
					horizon=5,
					act_dim=2,
					layer_norm=True,
					reuse=False):

		# decode the latent plan into actions
		with tf.variable_scope(scope, reuse=reuse):
			out = utT
			out = tf.reshape(out, [-1, latent_dim])
			for _ in range(n_hidden):
				if _ != n_hidden - 1:
					out = layers.fully_connected(out,
													num_outputs=latent_dim,
													activation_fn=None)
					if layer_norm is True:
						out = layers.layer_norm(out, center=True, scale=True)
					if nonlinearity == 'swish':
						out = tf.nn.sigmoid(out)*out
					if nonlinearity == 'relu':
						out = tf.nn.relu(out)
					if nonlinearity == 'tanh':
						out = tf.nn.tanh(out)
				else:
					out = layers.fully_connected(out,
													num_outputs=act_dim,
													activation_fn=None)
			out = tf.reshape(out, [-1, horizon, act_dim])
			return out

	def _sample_and_log_prob_zx(self,
									x,
									scope='plan_encoding',
									n_hidden=1,
									nonlinearity='swish',
									latent_dim=16,
									horizon=5,
									act_dim=2,
									layer_norm=True,
									reuse=False):

		# encode actions and sample from the latent space
		loc, scale = self._encode_plan(x,
									scope=scope,
									n_hidden=n_hidden,
									nonlinearity=nonlinearity,
									latent_dim=latent_dim,
									horizon=horizon,
									act_dim=act_dim,
									layer_norm=layer_norm,
									reuse=reuse)
		z_0 = tf.random_normal(tf.shape(loc), loc, scale)
		z_0.set_shape(loc.get_shape())
		log_qzx = log_normal(z_0, loc, scale)
		log_pz = log_normal(z_0, tf.zeros_like(z_0), tf.ones_like(z_0))
		return z_0, log_qzx, log_pz

	def _sample_and_log_prob_xz(self, 
								z,
								x,
								scope='plan_decoding',
								n_hidden=1,
								nonlinearity='swish',
								latent_dim=16,
								horizon=5,
								act_dim=2,
								layer_norm=True,
								reuse=False):

		# sample reconstructed actions from the decoder 
		x_ = self._decode_plan(z,
								scope=scope,
								n_hidden=n_hidden,
								nonlinearity=nonlinearity,
								latent_dim=latent_dim,
								horizon=horizon,
								act_dim=act_dim,
								layer_norm=layer_norm,
								reuse=reuse)
		log_pxz = log_bernoulli(x, x_)
		return x_, log_pxz

	def _fully_connected(self,
							x,
							n_hidden=1,
							scope='fully_connected',
							nonlinearity='swish',
							out_dim=16,
							layer_norm=True,
							reuse=False):

		with tf.variable_scope(scope, reuse=reuse):
			out = x
			for _ in range(n_hidden):
				out = layers.fully_connected(out,
												num_outputs=out_dim,
												activation_fn=None)
				if layer_norm is True:
					out = layers.layer_norm(out, center=True, scale=True)
				if nonlinearity == 'swish':
					out = tf.nn.sigmoid(out)*out
				elif nonlinearity == 'relu':
					out = tf.nn.relu(out)
			return out


	def _conv_variable(self, weight_shape):
		w = weight_shape[0]
		h = weight_shape[1]
		input_channels = weight_shape[2]
		output_channels = weight_shape[3]
		d = 1.0 / np.sqrt(input_channels * w * h)
		bias_shape = [output_channels]
		weight = tf.Variable(tf.random_uniform(weight_shape, minval=-d, maxval=d))
		bias   = tf.Variable(tf.random_uniform(bias_shape,   minval=-d, maxval=d))
		return weight, bias

	def train(self,
			il_lr_0,
			il_lr,
			ol_lr,
			sess):
		if not self.learn_lr:
			sess.run(self.train_op,
						feed_dict={ self.il_lr_0:il_lr_0,
									self.il_lr:il_lr,
									self.ol_lr:ol_lr})
		else:
			sess.run(self.train_op,
						feed_dict={ self.ol_lr:ol_lr})

	def stats(self,
			il_lr_0,
			il_lr,
			sess):
		if not self.learn_lr:
			vae_loss, kl_div_loss, bc_loss, plan_loss, xg_pred, xg, bc_loss_first_step, summ = sess.run([self.get_vae_loss_op,
																										self.get_kl_div_loss_op,
																										self.get_outer_loss_op,
																										self.get_inner_loss_op,
																										self.xg_pred,
																										self.get_xg,
																										self.get_outer_loss_first_step_op,
																										self.summ_op],
																										feed_dict={ self.il_lr_0:il_lr_0,
																													self.il_lr:il_lr})
		else:
			vae_loss, kl_div_loss, logprob_loss, bc_loss, plan_loss, xg_pred, xg, bc_loss_first_step, summ = sess.run([self.get_vae_loss_op,
																														self.get_kl_div_loss_op,
																														self.get_outer_loss_op,
																														self.get_inner_loss_op,
																														self.xg_pred,
																														self.get_xg,
																														self.get_outer_loss_first_step_op,
																														self.summ_op],
																														feed_dict={})

		return vae_loss, kl_div_loss, np.sqrt(bc_loss), np.sqrt(plan_loss), xg_pred, xg, np.sqrt(bc_loss_first_step), summ
