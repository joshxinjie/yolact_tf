import tensorflow as tf

from src.fpn import FeaturePyramidTopDownPath

class Yolact(tf.keras.Model):

    def __init__(self, num_fpn_output_filters, input_shape=(550, 550, 3)):
        super(Yolact, self).__init__()

        resnet_50_backbone = tf.keras.applications.ResNet50(
                                input_shape=input_shape,
                                include_top=False,
                                layers=tf.keras.layers,
                                weights='imagenet'
                            )

        c3 = resnet_50_backbone.layers[80].output
        c4 = resnet_50_backbone.layers[142].output
        c5 = resnet_50_backbone.layers[174].output

        self.yolact_resnet_50_backbone = tf.keras.Model(
                                            inputs=resnet_50_backbone.input,
                                            outputs=[c3, c4, c5]
                                        )
        self.fpn_top_down = FeaturePyramidTopDownPath(num_fpn_output_filters)
        

    def call(self, inputs):
        c3, c4, c5 = self.yolact_resnet_50_backbone(inputs)
        fpn_output = self.fpn_top_down(c3, c4, c5)
        
        return fpn_output