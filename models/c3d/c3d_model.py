import tensorflow.python.keras as keras

from tensorflow.python.keras.layers import Dense, Dropout, Conv3D, MaxPooling3D, Flatten
from tensorflow.python.keras.regularizers import l2


class C3D(keras.Model):

	def __init__(self,
	             num_classes=10,
	             weight_decay=5e-3,
	             input_shape=(112, 112, 16, 3),
	             dropout_ratio=0.5):
		super(C3D, self).__init__(name='C3D Model')
		self.num_classes = num_classes
		
		# layer definitions
		self.conv_1 = Conv3D(64, (3, 3, 3), strides=(1, 1, 1), padding='same', activation='relu', kernel_regularizer=l2(weight_decay))
		self.pool_1 = MaxPooling3D((2, 2, 1), strides=(2, 2, 1), padding='same')
		
		self.conv_2 = Conv3D(128, (3, 3, 3), strides=(1, 1, 1), padding='same', activation='relu', kernel_regularizer=l2(weight_decay))
		self.pool_2 = MaxPooling3D((2, 2, 2), strides=(2, 2, 2), padding='same')
		
		self.conv_3 = Conv3D(128, (3, 3, 3), strides=(1, 1, 1), padding='same', activation='relu', kernel_regularizer=l2(weight_decay))
		self.pool_3 = MaxPooling3D((2, 2, 2), strides=(2, 2, 2), padding='same')
		
		self.conv_4 = Conv3D(256, (3, 3, 3), strides=(1, 1, 1), padding='same', activation='relu', kernel_regularizer=l2(weight_decay))
		self.pool_4 = MaxPooling3D((2, 2, 2), strides=(2, 2, 2), padding='same')
		
		self.conv_4 = Conv3D(512, (3, 3, 3), strides=(1, 1, 1), padding='same', activation='relu', kernel_regularizer=l2(weight_decay))
		self.pool_4 = MaxPooling3D((2, 2, 2), strides=(2, 2, 2), padding='same')

		self.dense = Dense(2048, activation='relu', kernel_regularizer=l2(weight_decay))
		self.dropout = Dropout(dropout_ratio)
		self.dense_cls = Dense(num_classes, activation='softmax', kernel_regularizer=l2(weight_decay))

	def call(self, inputs, **kwargs):
		x = self.pool_1(self.conv_1(inputs))
		x = self.pool_2(self.conv_2(x))
		x = self.pool_3(self.conv_3(x))
		x = self.pool_4(self.conv_4(x))
		x = Flatten()(x)
		x = self.dropout(self.dense(x))
		x = self.dropout(self.dense(x))
		x = self.dense_cls(x)
		return x
	