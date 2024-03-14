# Import all of the dependencies
import streamlit as st
import os 
import tensorflow as tf
from PIL import Image
import numpy as np
from scipy.optimize import fmin_l_bfgs_b

from keras.models import Model
from keras.applications.vgg16 import VGG16


def artist(content_path, style_path, vgg_model, height = 512, width = 512):
    
    def compute_content_cost(content_output, generated_output):
        """
        Computes the content cost

        Arguments:
        a_C -- tensor of dimension (1, n_H, n_W, n_C), hidden layer activations representing content of the image C 
        a_G -- tensor of dimension (1, n_H, n_W, n_C), hidden layer activations representing content of the image G

        Returns: 
        J_content -- scalar that you compute using equation 1 above.
        """
        a_C = content_output[-1]
        a_G = generated_output[-1]


        # Retrieve dimensions from a_G (≈1 line)
        _, n_H, n_W, n_C = a_G.get_shape().as_list()

        # Reshape 'a_C' and 'a_G' (≈2 lines)
        # DO NOT reshape 'content_output' or 'generated_output'
        a_C_unrolled = tf.reshape(a_C, [n_H * n_W, n_C])
        a_G_unrolled = tf.reshape(a_G, [n_H * n_W, n_C])

        # compute the cost with tensorflow (≈1 line)
        J_content = tf.reduce_sum(tf.square(a_C_unrolled - a_G_unrolled)) # * 1 / (4 * n_H * n_W * n_C) * 

        return tf.convert_to_tensor(J_content)

    def compute_variation_loss(generated_image):
        a = tf.square(generated_image[:, :height-1, :width-1, :] - generated_image[:, 1:, :width-1, :])
        b = tf.square(generated_image[:, :height-1, :width-1, :] - generated_image[:, :height-1, 1:, :])
        J_variation = tf.reduce_sum(tf.pow(a + b, 1.25))
        return tf.convert_to_tensor(J_variation)
    
    def gram_matrix(A):
        """
        Argument:
        A -- matrix of shape (n_C, n_H*n_W)

        Returns:
        GA -- Gram matrix of A, of shape (n_C, n_C)
        """  
        GA = tf.linalg.matmul(A, tf.transpose(A))

        return tf.convert_to_tensor(GA)
    
    def compute_layer_style_cost(a_S, a_G):
        """
        Arguments:
        a_S -- tensor of dimension (1, n_H, n_W, n_C), hidden layer activations representing style of the image S 
        a_G -- tensor of dimension (1, n_H, n_W, n_C), hidden layer activations representing style of the image G

        Returns: 
        J_style_layer -- tensor representing a scalar value, style cost defined above by equation (2)
        """
        # Retrieve dimensions from a_G (≈1 line)
        _, n_H, n_W, n_C = a_G.get_shape().as_list()

        # Reshape the tensors from (1, n_H, n_W, n_C) to (n_C, n_H * n_W) (≈2 lines)
        a_S = tf.transpose(tf.reshape(a_S, [n_H * n_W, n_C]))
        a_G = tf.transpose(tf.reshape(a_G, [n_H * n_W, n_C]))

        # Computing gram_matrices for both images S and G (≈2 lines)
        GS = gram_matrix(a_S)
        GG = gram_matrix(a_G)

        # Computing the loss (≈1 line)
        J_style_layer = 1 / (4 * (n_H * n_W * n_C)**2) * tf.reduce_sum(tf.square(GS - GG))

        return tf.convert_to_tensor(J_style_layer)
    
    STYLE_LAYERS = [
    ('block1_conv2', 1.0),
    ('block2_conv2', 1.0),
    ('block3_conv3', 1.0),
    ('block4_conv3', 1.0),
    ('block5_conv3', 1.0)]
    
    def compute_style_cost(style_image_output, generated_image_output, STYLE_LAYERS=STYLE_LAYERS):
        """
        Computes the overall style cost from several chosen layers

        Arguments:
        style_image_output -- our tensorflow model
        generated_image_output --
        STYLE_LAYERS -- A python list containing:
                            - the names of the layers we would like to extract style from
                            - a coefficient for each of them

        Returns: 
        J_style -- tensor representing a scalar value, style cost defined above by equation (2)
        """

        # initialize the overall style cost
        J_style = 0

        # Set a_S to be the hidden layer activation from the layer we have selected.
        # The last element of the array contains the content layer image, which must not be used.
        a_S = style_image_output[:-1]

        # Set a_G to be the output of the choosen hidden layers.
        # The last element of the list contains the content layer image which must not be used.
        a_G = generated_image_output[:-1]
        for i, weight in zip(range(len(a_S)), STYLE_LAYERS):  
            # Compute style_cost for the current layer
            J_style_layer = compute_layer_style_cost(a_S[i], a_G[i])

            # Add weight * J_style_layer of this layer to overall style cost
            J_style += weight[1] * J_style_layer

        return tf.convert_to_tensor(J_style)
    
    def total_cost(J_content, J_style, J_variation,  content_weight = 0.025, style_weight = 1.0, total_variation_weight = 1):
        """
        Computes the total cost function

        Arguments:
        J_content -- content cost coded above
        J_style -- style cost coded above
        alpha -- hyperparameter weighting the importance of the content cost
        beta -- hyperparameter weighting the importance of the style cost

        Returns:
        J -- total cost as defined by the formula above.
        """

        #(≈1 line)
        J_content = tf.cast(J_content, tf.float64)
        J_style = tf.cast(J_style, tf.float64)
        J_variation = tf.cast(J_variation, tf.float64)
        content_weight = tf.cast(content_weight, tf.float64)
        style_weight = tf.cast(style_weight, tf.float64)
        total_variation_weight = tf.cast(total_variation_weight, tf.float64)

        J = J_content * content_weight + J_style * style_weight + J_variation * total_variation_weight
        J = J_content * content_weight + J_style * style_weight + J_variation * total_variation_weight

        return tf.convert_to_tensor(J)
    
    content_image = Image.open(content_path)
    #content_image = content_image.resize((height, width))
    
    style_image = Image.open(style_path)
    style_image = style_image.resize((width, height))
    
    content_array = np.array(content_image, dtype='float32')
    content_array = np.expand_dims(content_array, axis=0)
    
    style_array = np.array(style_image, dtype='float32')
    if style_array.ndim == 2:
        style_array = np.expand_dims(style_array, axis=-1)
        style_array = 255 - style_array
        style_array = np.broadcast_to(style_array, shape = (style_array.shape[0], style_array.shape[1], 3))
        
        content_image = content_image.convert("L")
        content_array = np.array(content_image, dtype='float32')
        content_array = np.expand_dims(content_array, axis=-1)
        content_array = 255 - content_array
        content_array = np.broadcast_to(content_array, shape = (content_array.shape[0], content_array.shape[1], 3))
        content_array = np.expand_dims(content_array.copy(), axis=0)
        
    style_array = np.expand_dims(style_array.copy(), axis=0)
    
    content_array[:, :, :, 0] -= 103.939
    content_array[:, :, :, 1] -= 116.779
    content_array[:, :, :, 2] -= 123.68
    content_array = content_array[:, :, :, ::-1]

    style_array[:, :, :, 0] -= 103.939
    style_array[:, :, :, 1] -= 116.779
    style_array[:, :, :, 2] -= 123.68
    style_array = style_array[:, :, :, ::-1]
    
    def get_layer_outputs(vgg, layer_names):
        """ Creates a vgg model that returns a list of intermediate output values."""
        outputs = [vgg.get_layer(layer[0]).output for layer in layer_names]

        model = tf.keras.Model([vgg.input], outputs)
        return model
    
    content_layer = [('block2_conv2', 1)]

    vgg_model_outputs = get_layer_outputs(vgg_model, STYLE_LAYERS + content_layer)
    
    vgg_model_outputs.trainable = False
    
    # Assign the content image to be the input of the VGG model.  
    # Set a_C to be the hidden layer activation from the layer we have selected
    preprocessed_content =  tf.Variable(content_array)
    a_C = vgg_model_outputs(preprocessed_content)
    
    # Assign the input of the model to be the "style" image 
    preprocessed_style =  tf.Variable(style_array)
    a_S = vgg_model_outputs(preprocessed_style)
    
    # Define the cost function
    def cost_function(generated_image):
        with tf.GradientTape() as tape:
            generated_image = tf.reshape(generated_image, (1, height, width, 3))
            a_G = vgg_model_outputs(generated_image) 
            J_style = compute_style_cost(a_S, a_G, STYLE_LAYERS=STYLE_LAYERS)
            J_variation = compute_variation_loss(generated_image)
            J_content = compute_content_cost(a_C, a_G)
            J = total_cost(J_content, J_style, J_variation,  content_weight = 0.025, style_weight = 5, total_variation_weight = 1)
        return J
    
    @tf.function()
    def gradient_function(generated_image):
        generated_image = tf.convert_to_tensor(generated_image)
        with tf.GradientTape() as tape:
            tape.watch(generated_image)
            generated_image = tf.reshape(generated_image, (1, height, width, 3))
            a_G = vgg_model_outputs(generated_image) 
            J_style = compute_style_cost(a_S, a_G, STYLE_LAYERS=STYLE_LAYERS)
            J_variation = compute_variation_loss(generated_image)
            J_content = compute_content_cost(a_C, a_G)
            J = total_cost(J_content, J_style, J_variation, content_weight=0.025, style_weight=50, total_variation_weight=1)
        grad = tape.gradient(J, generated_image)
        return tf.reshape(grad, [-1])
    
    # Define a function to print the cost after each epoch
    def print_cost(epoch, cost):
        print(f"Epoch: {epoch}, Cost: {cost}")
    
    x = np.random.uniform(0, 255, (1, height, width, 3)) - 128.
    
    image_placeholder = st.empty()
    for epoch in range(10):
        x, cost, _ = fmin_l_bfgs_b(cost_function, x, fprime=gradient_function, maxfun = 20)
        my_bar.progress((epoch+1) / 10, f'{progress_text} {epoch+1} out of 10')
        print_cost(epoch, cost)
        
        optimized_generated_image = tf.reshape(tf.identity(x), (height, width, 3))
        optimized_array = optimized_generated_image.numpy()
        
        optimized_array = optimized_array[:, :, ::-1]
        optimized_array[:, :, 0] += 103.939
        optimized_array[:, :, 1] += 116.779
        optimized_array[:, :, 2] += 123.68
        optimized_array = np.clip(optimized_array, 0, 255).astype('uint8')
        
        image_placeholder.image(optimized_array)
        
        if stop_button:
            global stop_button_check
            stop_button_check = True
            break
        
        
    optimized_generated_image = tf.reshape(x, (height, width, 3))
    optimized_array = optimized_generated_image.numpy()
    
    optimized_array = optimized_array[:, :, ::-1]
    optimized_array[:, :, 0] += 103.939
    optimized_array[:, :, 1] += 116.779
    optimized_array[:, :, 2] += 123.68
    optimized_array = np.clip(optimized_array, 0, 255).astype('uint8')
    
    return optimized_array




st.set_page_config(layout = "wide")

with st.sidebar:
    st.title("Artistic Style Transfer with Neural Networks")
    st.info('This application generates images by transferring the style of the art image to the content image')
    
st.title("Art")

content_options = os.listdir(os.path.join('..', 'images', 'content'))
content_options  = [file for file in content_options if file != ".DS_Store"]
selected_content = st.selectbox('Choose the content image', content_options)

style_options = os.listdir(os.path.join('..', 'images', 'style'))
style_options  = [file for file in style_options if file != ".DS_Store"]
selected_style = st.selectbox('Choose the art image', style_options)

col1, col2 = st.columns(2)

content_path = os.path.join('..', 'images', 'content', selected_content)
style_path = os.path.join('..', 'images', 'style', selected_style)


if style_path :
    with col1:
        st.info("This is the content image.")
        
        st.image(content_path)
        st.info("This is the art image.")
        st.image(style_path)
    
with col2:
    st.info("Press the button to generate image")
    generate_button = st.button("Generate")
    
    st.info("Press the button to stop training")
    stop_button = st.button("Stop")
    
    st.info("this is generated image")
    if generate_button:
        content_array = np.array(Image.open(content_path))
        width = content_array.shape[1]
        height = content_array.shape[0]
        
        vgg = tf.keras.applications.VGG16(include_top=False,
                                  input_shape=(height, width, 3),
                                  weights='imagenet')

        vgg.trainable = False
        
        progress_text = "Model training in progress. Please wait. Epoch: "
        my_bar = st.progress(0, text=progress_text)
        
        stop_button_check = None
        generated_image = artist(content_path, style_path, vgg_model = vgg, height = height, width = width)
        
        if stop_button_check:
            my_bar.progress(1.0, f"Training was interrupted ")
            st.image(generated_image)
        path = os.path.join('generated_images', selected_content)
        Image.fromarray(generated_image).save(path)
        
    