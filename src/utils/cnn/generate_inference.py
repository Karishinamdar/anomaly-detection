import os
from models import *
from generator import *
import cv2

# from models import *
IMAGE_NAME = 'LC08_L1TP_120043_20200830_20200830_01_RT_p00417.tif'

IMAGE_PATH = '../../../dataset/images/patches/'
OUTPUT_PATH = './output'

# 0 or 1
CUDA_DEVICE = 0

# Select channels 10 or 3
N_CHANNELS = 3
# Select filters 16 or 64
N_FILTERS = 16

MASK_ALGORITHM = 'Murphy'
MODEL_NAME = 'unet'
IMAGE_SIZE = (256, 256)
TH_FIRE = 0.25

# 
MODEL_FOLDER_NAME = 'murphy' if MASK_ALGORITHM == 'Murphy' else MASK_ALGORITHM
ARCHITECTURE = '{}_{}f_2conv_{}'.format( MODEL_NAME, N_FILTERS, '762' if N_CHANNELS == 3 else '762' )

WEIGHTS_FILE = '../../train/{}/{}/weights/model_{}_{}_final_weights.h5'.format(MODEL_FOLDER_NAME, ARCHITECTURE, MODEL_NAME, MASK_ALGORITHM)

print(ARCHITECTURE)
print(WEIGHTS_FILE)

if not os.path.exists(OUTPUT_PATH):
    os.makedirs(OUTPUT_PATH)

os.environ["CUDA_VISIBLE_DEVICES"] = str(CUDA_DEVICE)
model = get_model(MODEL_NAME, input_height=IMAGE_SIZE[0], input_width=IMAGE_SIZE[1], n_filters=N_FILTERS, n_channels=N_CHANNELS)
model.summary()

print('Loading weghts...')
model.load_weights(WEIGHTS_FILE)
print('Weights Loaded')

img_path = os.path.join(IMAGE_PATH, IMAGE_NAME)

print('IMAGE: {}'.format(img_path))

img = get_img_arr(img_path)

x = np.array([img])
print(f"x.shape={x.shape}")
y_pred = model.predict(x, batch_size=1)
y_pred = y_pred[0, :, :, 0] > TH_FIRE

y_pred = np.array(y_pred * 255, dtype=np.uint8)

output_image = os.path.join(OUTPUT_PATH, IMAGE_NAME)
cv2.imwrite(output_image, cv2.cvtColor(y_pred, cv2.COLOR_RGB2BGR))

print('Done!')
