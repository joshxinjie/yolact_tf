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

        c3 = resnet_50_backbone.layers[80].output
        c4 = resnet_50_backbone.layers[142].output
        c5 = resnet_50_backbone.layers[174].output

        yolact_resnet_50_backbone = tf.keras.Model(
                                        inputs=resnet_50_backbone.input,
                                        outputs=[c3, c4, c5]
                                    )

        

    def call(self, inputs):
        c3, c4, c5 = self.backbone_resnet(inputs)