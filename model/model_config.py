TYPE = 'swinv2'

MODE = 'l' # 'b', 's'

CONF ={
    'IN_CHANS' : 3,
    'PATCH_SIZE' : 4,
    'MLP_RATIO' : 4.,
    'QKV_BIAS' : True,
    'QK_SCALE' : None,
    'APE' : False,
    'PATCH_NORM' : True,
    'PRETRAINED_WINDOW_SIZES' : [12,12,12,6],
    'DROP_RATE' : 0.0,
    'FUSED_LAYERNORM': False,
}

# IMG_SIZE = (192,640)정석, (320,640) 반반 (256,768)
MODEL = {
    # 's': {'IMG_SIZE':(352,1056),'EMBED_DIM':96,'DEPTHS':[2,2,18,2],'NUM_HEADS':[3,6,12,24],'WINDOW_SIZE':11,'DROP_PATH_RATE':0.3}, #window = 16
    's': {'IMG_SIZE':(352,1120),'EMBED_DIM':96,'DEPTHS':[2,2,18,2],'NUM_HEADS':[3,6,12,24],'WINDOW_SIZE':[16,16,16,11], 'DROP_PATH_RATE':0.3},
    'b': {'IMG_SIZE':(352,1120),'EMBED_DIM':128,'DEPTHS':[2,2,18,2],'NUM_HEADS':[4,8,16,32],'WINDOW_SIZE':[16,16,16,11],'DROP_PATH_RATE':0.3}, #window = 12
    # 'b': {'IMG_SIZE':(256,768),'EMBED_DIM':128,'DEPTHS':[2,2,18,2],'NUM_HEADS':[4,8,16,32],'WINDOW_SIZE':12,'DROP_PATH_RATE':0.2},
    'l': {'IMG_SIZE':(352,1120),'EMBED_DIM':192,'DEPTHS':[2,2,18,2],'NUM_HEADS':[6,12,24,48],'WINDOW_SIZE':[16,16,16,11],'DROP_PATH_RATE':0.3},
}

PRETRAIN = {
    's': '/home/mvpcoin/lee/depthmodel/pixelformer/model/pretrained/swinv2_small_patch4_window16_256.pth',
    'b': '/home/mvpcoin/lee/depthmodel/pixelformer/model/pretrained/swin_v2_base_simmim.pth',
    'l': '/home/mvpcoin/lee/depthmodel/pixelformer/model/pretrained/swin_v2_large_simmim.pth'
}