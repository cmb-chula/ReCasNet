import tensorflow as tf

def get_backbone(input_shape = (128, 128, 3), backbone_name = 'effnet'):
    backbone = None
    print(backbone_name)
    if(backbone_name == 'effnet'):
        import sys
        import efficientnet.tfkeras as efn
        backbone = efn.EfficientNetB4(input_shape = input_shape, weights='imagenet', include_top = False)
    elif(backbone_name == 'resnet'):
        from tensorflow.keras.applications.resnet import ResNet152
        backbone = ResNet152(input_shape = input_shape, weights='imagenet', include_top = False)
    elif(backbone_name == 'densenet'):
        from tensorflow.keras.applications.densenet import DenseNet169
        backbone = DenseNet169(input_shape = input_shape, weights='imagenet', include_top = False)
    elif(backbone_name == 'convnext'):
        from tensorflow.keras.applications.convnext import ConvNeXtBase
        backbone = ConvNeXtBase(input_shape = input_shape, weights='imagenet', include_top = False)
    elif(backbone_name == 'convnexts'):
        from tensorflow.keras.applications.convnext import ConvNeXtSmall
        backbone = ConvNeXtSmall(input_shape = input_shape, weights='imagenet', include_top = False)

    return backbone

def get_classification_model(backbone, input_shape = (250, 100, 3), num_classes = 5, return_embedding = False, visualize = False):
    pooler = tf.keras.layers.GlobalAveragePooling2D()(backbone.output)
    out = tf.keras.layers.Dense(num_classes)(pooler)
    softmax = tf.keras.layers.Activation('softmax')(out)
    if(return_embedding):
        model = tf.keras.models.Model(inputs = backbone.input, outputs =  pooler)
        return model
    elif(visualize):
        model = tf.keras.models.Model(inputs = backbone.input, outputs =  [backbone.output, softmax])
        return model
    else:
        model = tf.keras.models.Model(inputs = backbone.input, outputs = softmax)
        return model

def get_relocation_model(backbone, input_shape = (250, 100, 3), num_classes = 5, return_embedding = False):
    pooler = tf.keras.layers.GlobalAveragePooling2D()(backbone.output)
    out = tf.keras.layers.Dense(2)(pooler)
    # out = tf.keras.layers.Activation('tanh')(out)
    if(return_embedding):
        model = tf.keras.models.Model(inputs = backbone.input, outputs =  pooler)
        return model
    else:
        model = tf.keras.models.Model(inputs = backbone.input, outputs = out)
        return model
    return model

def get_relocation_aux_model(backbone, input_shape = (250, 100, 3), num_classes = 5, return_embedding = False):
    pooler = tf.keras.layers.GlobalAveragePooling2D()(backbone.output)
    out1 = tf.keras.layers.Dense(2)(pooler)
    out2 = tf.keras.layers.Dense(num_classes, activation = 'softmax')(pooler)

    # out = tf.keras.layers.Activation('tanh')(out)
    model = tf.keras.models.Model(inputs = backbone.input, outputs = [out1, out2])
    return model




def efficientnetb4_teacher_soft_softmax(input_shape = (250, 100, 3), num_classes = 5, c = 10):
    # 285, 120 2.44 | 414, 163, 2.65
    import efficientnet.tfkeras as efn
    backbone = efn.EfficientNetB4(input_shape = input_shape, weights='imagenet', include_top = False)
    pooler = tf.keras.layers.GlobalAveragePooling2D()(backbone.output)
    out = tf.keras.layers.Dense(num_classes)(pooler)
    softer = tf.keras.layers.Lambda(lambda x : x / c)(out)
    softmax = tf.keras.layers.Activation('softmax')(out)
    model = tf.keras.models.Model(inputs = backbone.input, outputs = softmax)
    return model

def efficientnetb4_distillation(input_shape = (250, 100, 3), num_classes = 5, c = 5):
    # 285, 120 2.44 | 414, 163, 2.65
    import efficientnet.tfkeras as efn
    backbone = efn.EfficientNetB4(input_shape = input_shape, weights='imagenet', include_top = False)
    pooler = tf.keras.layers.GlobalAveragePooling2D()(backbone.output)
    logit = tf.keras.layers.Dense(num_classes)(pooler)
    hard_softmax = tf.keras.layers.Activation('softmax', name = 'hard_softmax')(logit)

    logits_soft = tf.keras.layers.Lambda(lambda x : x / c)(logit)
    soft_softmax = tf.keras.layers.Activation('softmax', name = 'soft_softmax')(logits_soft)
    model = tf.keras.models.Model(inputs = backbone.input, outputs = [hard_softmax, soft_softmax])
    return model

def efficientnetb4_teacher_multitask(input_shape = (250, 100, 3), num_classes = [5, 5]):
    # 285, 120 2.44 | 414, 163, 2.65
    import efficientnet.tfkeras as efn
    backbone = efn.EfficientNetB4(input_shape = input_shape, weights='imagenet', include_top = False)
    pooler = tf.keras.layers.GlobalAveragePooling2D()(backbone.output)
    out = tf.keras.layers.Dense(num_classes[0])(pooler)
    softmax = tf.keras.layers.Activation('softmax', name = 'ethnic')(out)
    out2 = tf.keras.layers.Dense(512)(pooler)
    out2 = tf.keras.layers.BatchNormalization()(out2)
    out2 = tf.keras.layers.Dense(num_classes[1])(pooler)
    softmax2 = tf.keras.layers.Activation('softmax', name = 'age')(out2)

    model = tf.keras.models.Model(inputs = backbone.input, outputs = [softmax, softmax2])
    return model

def efficientnetb4_teacher_transfer(pretrain_path, input_shape = (250, 100, 3), num_classes = 5):
    # 285, 120 2.44 | 414, 163, 2.65
    import efficientnet.tfkeras as efn
    base_model = efficientnetb4_teacher(input_shape, num_classes = 5)
    base_model.load_weights(pretrain_path)
    # out = tf.keras.layers.Dense(num_classes)(base_model.get_layer(index=-3).output)
    # softmax = tf.keras.layers.Activation('softmax')(out)
    # model = tf.keras.models.Model(inputs = base_model.input, outputs = softmax)
    # model.summary()
    # return model
    return base_model

def efficientnetb4_student(input_shape = (250, 100, 3), num_classes = 5):
    # 285, 120 2.44 | 414, 163, 2.65
    import efficientnet.tfkeras as efn
    backbone = efn.EfficientNetB4(input_shape = input_shape, weights='imagenet', include_top = False)
    pooler = tf.keras.layers.GlobalAveragePooling2D()(backbone.output)
    drop = tf.keras.layers.Dropout(0.5)(pooler)
    out = tf.keras.layers.Dense(num_classes)(drop)
    softmax = tf.keras.layers.Activation('softmax')(out)

    model = tf.keras.models.Model(inputs = backbone.input, outputs = softmax)
    return model

def resnet50_teacher(input_shape = (250, 100, 3), num_classes = 5):
    # 285, 120 2.44 | 414, 163, 2.65
    from tensorflow.keras.applications.resnet_v2 import ResNet50V2
    backbone = ResNet50V2(input_shape = input_shape, weights='imagenet', include_top = False)
    pooler = tf.keras.layers.GlobalAveragePooling2D()(backbone.output)
    out = tf.keras.layers.Dense(num_classes)(pooler)
    softmax = tf.keras.layers.Activation('softmax')(out)

    model = tf.keras.models.Model(inputs = backbone.input, outputs = softmax)
    return model

def B2block(inp):
    x = tf.keras.layers.Conv2D(512, (1, 1), padding = 'same')(inp)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.Conv2D(512, (3, 3), padding = 'same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.Conv2D(512, (1, 1), padding = 'same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.Concatenate()([x, inp])
    return x
    
def RACNN(input_shape = (250, 100, 3), num_classes = 4, return_embedding = False):
    # 285, 120 2.44 | 414, 163, 2.65
    import sys
    sys.path.append('model/')
    import efficientnet.tfkeras as efn
    inp1 = tf.keras.layers.Input(shape = (4,4,2048))
    inp2 = tf.keras.layers.Input(shape = (1792))
    # att1 = tf.keras.layers.GlobalAveragePooling2D()(inp1)
    # att1 = tf.keras.layers.Dense(int(1792 // 16))(att1)
    # att1 = tf.keras.layers.Activation('relu')(att1)
    # att1 = tf.keras.layers.Dense(1792)(att1)
    # att1 = tf.keras.layers.Activation('relu')(att1)
    # att1 = tf.reshape(att1, (-1, 1, 1, 1792))
    # x = tf.keras.layers.Multiply()([inp1, att1])
    # x = tf.keras.layers.Add()([inp1, x])

    att1 = tf.math.reduce_mean(inp1, axis = -1)[..., tf.newaxis]
    att2 = tf.math.reduce_max(inp1, axis = -1)[..., tf.newaxis]
    att = tf.keras.layers.Concatenate()([att1, att2])
    att = tf.keras.layers.Conv2D(1, (1,1), padding = 'same')(att)
    att = tf.keras.layers.Activation('sigmoid')(att)
    x = tf.keras.layers.Multiply()([inp1, att])

    for _ in range(3):
        x = B2block(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Concatenate()([x, inp2])
    # x = tf.keras.layers.Dense(2048)(x)
    # x = tf.keras.layers.Activation('relu')(x)

    out = tf.keras.layers.Dense(num_classes)(x)
    softmax = tf.keras.layers.Activation('softmax')(out)

    model = tf.keras.models.Model(inputs = [inp1, inp2], outputs =  softmax)
    return model
    # x = 

    # backbone = efn.EfficientNetB4(input_shape = input_shape, weights='imagenet', include_top = False)
    # pooler = tf.keras.layers.GlobalAveragePooling2D()(backbone.output)
    # out = tf.keras.layers.Dense(num_classes)(pooler)
    # softmax = tf.keras.layers.Activation('softmax')(out)
    
    # if(return_embedding):
    #     model = tf.keras.models.Model(inputs = backbone.input, outputs =  pooler)
    #     return model
    # else:
    #     model = tf.keras.models.Model(inputs = backbone.input, outputs = softmax)
    #     return model