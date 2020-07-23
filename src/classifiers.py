import tensorflow as tf


class SimpleConv2DClassifier(tf.keras.Model):
	"""
	# Arguments
		num_classes: number of classes of your problem
	#### Usage: Use it as a keras model, just compile it with categorical_crossentropy and fit it on the data.  
	"""
	def __init__(self, num_classes):
		super(SimpleConv2DClassifier, self).__init__()
		self.conv_inp = tf.keras.layers.Conv2D(32, (3, 3), activation='relu')
		self.conv_64 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu')
		self.maxP = tf.keras.layers.MaxPooling2D((2, 2))
		self.conv_128 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu')
		self.conv_256 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu')
		self.flat = tf.keras.layers.Flatten()
		self.FC_l1 = tf.keras.layers.Dense(128, activation='relu')
		self.drop = tf.keras.layers.Dropout(0.5) 
		self.FC_l2 = tf.keras.layers.Dense(64, activation='relu')
		self.FC_l3 = tf.keras.layers.Dense(32, activation='relu')
		self.classifier = tf.keras.layers.Dense(num_classes, activation='softmax')

		
	def call(self, inputs):
		x = self.conv_inp(inputs)
		x = self.maxP(x)
		x = self.conv_64(x)
		x = self.maxP(x)
		x = self.conv_128(x)
		x = self.maxP(x)
		x = self.conv_256(x)
		x = self.maxP(x)
		x = self.flat(x)
		x = self.FC_l1(x)
		x = self.drop(x)
		x = self.FC_l2(x)
		x = self.FC_l3(x)
		x = self.classifier(x)
		return x


def create_model(num_classes): 
	
	image_input = tf.keras.Input(shape=(120, 120, 3), name='image_input')
	meta_input = tf.keras.Input(shape=(486,), name='meta_input')

	x = tf.keras.layers.Conv2D(32, (3, 3), activation='relu')(image_input)
	x = tf.keras.layers.MaxPooling2D((2, 2))(x)
	x = tf.keras.layers.Conv2D(64, (3, 3), activation='relu')(x)
	x = tf.keras.layers.MaxPooling2D((2, 2))(x)
	x = tf.keras.layers.Conv2D(128, (3, 3), activation='relu')(x)
	x = tf.keras.layers.MaxPooling2D((2, 2))(x)
	x = tf.keras.layers.Conv2D(256, (3, 3), activation='relu')(x)
	x = tf.keras.layers.MaxPooling2D((2, 2))(x)
	x = tf.keras.layers.Flatten()(x)
	x = tf.keras.layers.concatenate([x, meta_input])
	x = tf.keras.layers.Dense(128, activation='relu')(x)
	x = tf.keras.layers.Dropout(0.5)(x)
	x = tf.keras.layers.Dense(64, activation='relu')(x) 
	x = tf.keras.layers.Dense(32, activation='relu')(x)
	x = tf.keras.layers.Dense(num_classes, activation='softmax')(x)

	model = tf.keras.models.Model(inputs=[image_input, meta_input], outputs=[x])


	return model

def SimpleDenseLayersClassifier(num_classes):
	num_input = tf.keras.Input(shape=(486,), name='num_input')
	x = tf.keras.layers.Dense(256, activation='relu')(num_input)
	x = tf.keras.layers.Dropout(0.5)(x)
	x = tf.keras.layers.Dense(128, activation='relu')(x)
	x = tf.keras.layers.Dropout(0.5)(x)
	x = tf.keras.layers.Dense(64, activation='relu')(x)
	x = tf.keras.layers.Dropout(0.5)(x)
	x = tf.keras.layers.Dense(32, activation='relu')(x)
	x = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
	model = tf.keras.models.Model(inputs=[num_input], outputs=[x])

	return model