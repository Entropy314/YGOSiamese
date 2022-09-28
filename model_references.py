import tensorflow.keras.applications as tfk

WEIGHTS = {'IMAGENET': 'imagenet'}

MODEL = {
         'RESNET50V2': tfk.resnet_v2.ResNet50V2, 
         'RESNET101V2': tfk.resnet_v2.ResNet101V2, 
         'RESNET152V2': tfk.resnet_v2.ResNet152V2, 
         'RESNET50': tfk.resnet.ResNet50, 
         'RESNET101': tfk.resnet.ResNet101, 
         'RESNET152': tfk.resnet.ResNet152, 
         'EFFICIENTNETV2L': tfk.efficientnet_v2.EfficientNetV2L,
         'EFFICIENTNETV2M': tfk.efficientnet_v2.EfficientNetV2M,
         'EFFICIENTNETV2S': tfk.efficientnet_v2.EfficientNetV2S,
         'MOBILENET': tfk.mobilenet.MobileNet,
         'MOBILENETV2': tfk.mobilenet_v2.MobileNetV2
         }

PREPROCESS = {
                'RESNET50V2': tfk.resnet_v2, 
                'RESNET101V2': tfk.resnet_v2, 
                'RESNET152V2': tfk.resnet_v2, 
                'RESNET50': tfk.resnet, 
                'RESNET101': tfk.resnet, 
                'RESNET152': tfk.resnet, 
                'EFFICIENTNETV2L': tfk.efficientnet_v2,
                'EFFICIENTNETV2M': tfk.efficientnet_v2,
                'EFFICIENTNETV2S': tfk.efficientnet_v2,
                'MOBILENET': tfk.mobilenet,
                'MOBILENETV2': tfk.mobilenet_v2
             }   
