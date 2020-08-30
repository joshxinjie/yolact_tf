import tensorflow as tf

class FeaturePyramidTopDownPath(tf.keras.layers.Layer):
    """
    FPN paper reference: https://arxiv.org/pdf/1612.03144.pdf
    Suggested upsampling by a factor of 2. Did not mention details
    about cropping the upsampled feature maps to enable elementwise
    adding. Perhaps in original FPN, upsampled feature maps are already
    of correct size.
    
    """
    def __init__(self, num_fpn_output_filters=256):
        super(FeaturePyramidTopDownPath, self).__init__()
        
        # use 1x1 conv to reduce num of channels of c5
        # need padding=same?
        self.c5_1by1_conv_layer = tf.keras.layers.Conv2D(filters=num_fpn_output_filters, kernel_size=(1,1))
        # use 1x1 conv to reduce num of channels of c4
        self.c4_1by1_conv_layer = tf.keras.layers.Conv2D(filters=num_fpn_output_filters, kernel_size=(1,1))
        # apply 1x1 conv to reduce num of channels of c3
        self.c3_1by1_conv_layer = tf.keras.layers.Conv2D(filters=num_fpn_output_filters, kernel_size=(1,1))
        
        # upsample by a facor of 2
        # no parameters to learn, so one layer sufficient for both c4 and c5
        self.upsample_layer =  tf.keras.layers.UpSampling2D(size=(2, 2), interpolation='bilinear')
        
        # apply 3×3 convolution on each merged map togenerate the final feature map
        self.p5_3by3_conv_layer = tf.keras.layers.Conv2D(filters=num_fpn_output_filters, kernel_size=(3,3))
        self.p4_3by3_conv_layer = tf.keras.layers.Conv2D(filters=num_fpn_output_filters, kernel_size=(3,3))
        self.p3_3by3_conv_layer = tf.keras.layers.Conv2D(filters=num_fpn_output_filters, kernel_size=(3,3))
        
        # paper mention using 3x3 conv and stride of 2 to downsample p5 to get p6
        # likewise for p7
        self.downsample_p5_layer = tf.keras.layers.Conv2D(filters=num_fpn_output_filters, kernel_size=(3,3), strides=(2, 2))
        self.downsample_p6_layer = tf.keras.layers.Conv2D(filters=num_fpn_output_filters, kernel_size=(3,3), strides=(2, 2))
        
    def call(self, c3, c4, c5):
        c5_reduced_dim = self.c5_1by1_conv_layer(c5)
        upsampled_c5_reduced_dim = self.upsample_layer(c5_reduced_dim)
        
        c4_reduced_dim = self.c4_1by1_conv_layer(c4)
        upsampled_c4_reduced_dim = self.upsample_layer(c4_reduced_dim)
        
        c3_reduced_dim = self.c5_1by1_conv_layer(c3)
        
        p5 = self.p5_3by3_conv_layer(c5_reduced_dim)
        
        c4_add_c5 = self.crop_and_add(upsampled_c5_reduced_dim, c4_reduced_dim)
        p4 = self.p4_3by3_conv_layer(c4_add_c5)
        
        c3_add_c4 = self.crop_and_add(upsampled_c4_reduced_dim, c3_reduced_dim)
        p3 = self.p3_3by3_conv_layer(c3_add_c4)
        
        p6 = self.downsample_p5_layer(p5)
        p7 = self.downsample_p6_layer(p6)
        
        return [p3, p4, p5, p6, p7]
        
    def crop_and_add(self, x1, x2):
        """
        Follow TF implementation, where for example p5 is upsampled by twice and then a middle portion of it 
        that matches p4 is extracted and added to p4.
        
        This is how UNET does it
        https://tf-unet.readthedocs.io/en/latest/_modules/tf_unet/layers.html
        
        Will it be better to just resize p5 to the same size as p4? Less loss of information? But have to handle
        upsampling by float numbers?
        """
    
        x1_shape = x1.shape
        x2_shape = x2.shape
        #x1_shape = tf.shape(x1)
        #x2_shape = tf.shape(x2)
        
        # offsets for the top left corner of the crop
        # find the indices where the crop of x1 should be taken
        # with starting indices from the top left corner
        offsets = [0, (x1_shape[1] - x2_shape[1]) // 2, (x1_shape[2] - x2_shape[2]) // 2, 0]
        # the final size that x1 should be cropped into
        x1_cropped_size = [-1, x2_shape[1], x2_shape[2], -1]
        x1_crop = tf.slice(x1, offsets, x1_cropped_size)
        return tf.add(x1_crop, x2)