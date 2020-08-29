import tensorflow as tf

class Yolact(tf.keras.Model):

    def __init__(self):
        super(Yolact, self).__init__()

        resnet_50_backbone = tf.keras.applications.ResNet50(
                                input_shape=(550, 550, 3),
                                include_top=False,
                                layers=tf.keras.layers,
                                weights='imagenet'
                            )

        conv3_block4_out = resnet_50_backbone.layers[80].output
        conv4_block6_out = resnet_50_backbone.layers[142].output
        conv4_block6_out = resnet_50_backbone.layers[174].output

        yolact_resnet_50_backbone = tf.keras.Model(
                                        inputs=resnet_50_backbone.input,
                                        outputs=[conv3_block4_out, conv4_block6_out, conv4_block6_out
                                    ])

    def call():
        pass