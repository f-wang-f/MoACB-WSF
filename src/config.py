import os

# ==================== 数据相关配置 ====================
DATA_FILENAME = 'winddata.xlsx'
FEATURE_COLUMNS = ['Wind Direction', 'Theoretical_Power_Curve (KWh)', 'LV ActivePower (kW)', 'Wind Speed (m/s)']
TARGET_COLUMN = 'Wind Speed (m/s)'
SEQUENCE_LENGTH = 20
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
MIN_SAMPLES = 100

# ==================== NSGA-II 算法配置 ====================
POP_SIZE = 80
MAX_GEN = 40
NUM_RUNS = 1
MUTATION_PROB = 0.5
CROSSOVER_PROB = 0.8
LAMBDA_REG = 1e-4

# ==================== 模型训练配置 ====================
EVAL_EPOCHS = 20
EVAL_PATIENCE = 5
FINAL_EPOCHS = 200
FINAL_PATIENCE = 20
GRADIENT_CLIP = 1.0
MAX_MODEL_SIZE = 1e7

# ==================== 参数编码边界 ====================
KNUM_RANGE = (0, 7)
KSIZE_RANGE = (0, 3)
KACT_RANGE = (0, 7)
PT_RANGE = (0, 2)
PS_RANGE = (0, 3)
BS_RANGE = (0, 3)
OPT_RANGE = (0, 3)
LR_RANGE = (0.0001, 0.01)
REG_RANGE = (0, 3)

# ==================== 网络拓扑配置 ====================
NUM_MODULES = 5
NUM_CNN_MODULES = 3
NUM_LSTM_MODULES = 2
TOPO_BITS_LENGTH = NUM_MODULES * (NUM_MODULES - 1) // 2
INDIVIDUAL_LENGTH = TOPO_BITS_LENGTH + 5 * NUM_CNN_MODULES + 5 * NUM_LSTM_MODULES + 4

# ==================== 激活函数映射 ====================
ACTIVATION_MAP = {
    0: 'Softplus',
    1: 'Softsign',
    2: 'ELU',
    3: 'Softmax',
    4: 'Sigmoid',
    5: 'Tanh',
    6: 'ReLU',
    7: 'Identity'
}

# ==================== 优化器映射 ====================
OPTIMIZER_MAP = {
    0: 'SGD',
    1: 'Adam',
    2: 'AdaDelta',
    3: 'RMSprop'
}

# ==================== 正则化映射 ====================
REGULARIZER_MAP = {
    0: None,
    1: 'L1',
    2: 'L2',
    3: 'L1L2'
}

# ==================== 批大小映射 ====================
BATCH_SIZE_MAP = {
    0: 32,
    1: 64,
    2: 96,
    3: 128
}

# ==================== 输出目录 ====================
OUTPUT_DIR = 'output'
os.makedirs(OUTPUT_DIR, exist_ok=True)