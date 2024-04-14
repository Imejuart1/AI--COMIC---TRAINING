import tensorflow as tf
import numpy as np
import os
import cv2

# Define the generator model
def build_generator(input_shape):
    inputs = tf.keras.layers.Input(shape=input_shape)
    
    # Encoder
    x = tf.keras.layers.Conv2D(64, kernel_size=3, strides=2, padding='same')(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
    
    # Bottleneck
    x = tf.keras.layers.Conv2D(128, kernel_size=3, strides=2, padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
    
    # Decoder
    x = tf.keras.layers.Conv2DTranspose(64, kernel_size=3, strides=2, padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
    
    # Output
    outputs = tf.keras.layers.Conv2DTranspose(1, kernel_size=3, strides=2, padding='same', activation='sigmoid')(x)
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model

# Define the discriminator model
def build_discriminator(input_shape):
    inputs = tf.keras.layers.Input(shape=input_shape)
    
    # Encoder
    x = tf.keras.layers.Conv2D(64, kernel_size=3, strides=2, padding='same')(inputs)
    x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
    
    # Bottleneck
    x = tf.keras.layers.Conv2D(128, kernel_size=3, strides=2, padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
    
    # Output
    outputs = tf.keras.layers.Conv2D(1, kernel_size=3, strides=2, padding='same', activation='sigmoid')(x)
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model

# Load and preprocess dataset
def load_dataset(data_dir):
    sketchy_images = []
    thicker_images = []

    for file_name in os.listdir(data_dir):
        if file_name.endswith('.jpg') or file_name.endswith('.png'):
            # Load sketchy image
            sketchy_imagepath = os.path.join(data_dir, file_name)
            sketchy_img = cv2.imread(sketchy_imagepath, cv2.IMREAD_GRAYSCALE)
            # Load thicker image
            thicker_imagepath = os.path.join(data_dir, 'thicker_' + file_name)  # Assuming thicker images have "thicker_" prefix
            thicker_img = cv2.imread(thicker_imagepath, cv2.IMREAD_GRAYSCALE)

            if sketchy_img is not None and thicker_img is not None:
                sketchy_images.append(sketchy_img)
                thicker_images.append(thicker_img)

    # Convert lists to numpy arrays
    sketchy_images = np.array(sketchy_images)
    thicker_images = np.array(thicker_images)

    return sketchy_images, thicker_images

def preprocess_images(images):
    # Resize images to a common size (e.g., 256x256)
    resized_images = [cv2.resize(img, (256, 256)) for img in images]

    # Normalize pixel values to range [0, 1]
    normalized_images = np.array(resized_images) / 255.0

    return normalized_images

# Define loss functions
cross_entropy = tf.keras.losses.BinaryCrossentropy()

def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)

def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    return real_loss + fake_loss

# Set up optimizers
generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

# Define training step
@tf.function
def train_step(sketchy_images, thicker_images):
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(sketchy_images, training=True)

        real_output = discriminator(thicker_images, training=True)
        fake_output = discriminator(generated_images, training=True)

        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

# Load and preprocess dataset
data_dir = r'C:\Users\USER\Documents\AI- COMIC - TRAINING\IMAGES-AI'
sketchy_images, thicker_images = load_dataset(data_dir)
sketchy_images = preprocess_images(sketchy_images)
thicker_images = preprocess_images(thicker_images)

# Get input shape
input_shape = sketchy_images.shape[1:]

# Build generator and discriminator models
generator = build_generator(input_shape)
discriminator = build_discriminator(input_shape)

# Train the model
num_epochs = 100
batch_size = 32

for epoch in range(num_epochs):
    for i in range(0, len(sketchy_images), batch_size):
        sketchy_batch = sketchy_images[i:i+batch_size]
        thicker_batch = thicker_images[i:i+batch_size]
        gen_loss, disc_loss = train_step(sketchy_batch, thicker_batch)  # Capture loss values
    print(f'Epoch {epoch+1}/{num_epochs}, Generator Loss: {gen_loss}, Discriminator Loss: {disc_loss}')

# Save the trained generator model
generator.save('generator_model.h5')

