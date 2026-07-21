import numpy as np

from .config import (
    NUM_MODULES, NUM_CNN_MODULES, NUM_LSTM_MODULES, TOPO_BITS_LENGTH, INDIVIDUAL_LENGTH,
    KNUM_RANGE, KSIZE_RANGE, KACT_RANGE, PT_RANGE, PS_RANGE,
    BS_RANGE, OPT_RANGE, LR_RANGE, REG_RANGE,
    OPTIMIZER_MAP, REGULARIZER_MAP, BATCH_SIZE_MAP
)


def encode_individual(topo, cnn_params, lstm_params, setting):
    individual = topo + [item for sublist in cnn_params for item in sublist] + \
                 [item for sublist in lstm_params for item in sublist] + setting
    return np.array(individual, dtype=float)


def decode_individual(individual):
    idx = 0
    n = NUM_MODULES
    bits_length = TOPO_BITS_LENGTH
    topo = [int(round(individual[idx + i])) for i in range(bits_length)]
    idx += bits_length
    
    cnn_params = []
    for i in range(NUM_CNN_MODULES):
        knum = int(round(individual[idx]))
        ksize = int(round(individual[idx + 1]))
        kact = int(round(individual[idx + 2]))
        pt = int(round(individual[idx + 3]))
        ps = int(round(individual[idx + 4]))
        knum = max(KNUM_RANGE[0], min(KNUM_RANGE[1], knum))
        ksize = max(KSIZE_RANGE[0], min(KSIZE_RANGE[1], ksize))
        kact = max(KACT_RANGE[0], min(KACT_RANGE[1], kact))
        pt = max(PT_RANGE[0], min(PT_RANGE[1], pt))
        ps = max(PS_RANGE[0], min(PS_RANGE[1], ps))
        cnn_params.append([knum, ksize, kact, pt, ps])
        idx += 5
    
    lstm_params = []
    for i in range(NUM_LSTM_MODULES):
        knum = int(round(individual[idx]))
        lnum = int(round(individual[idx + 1]))
        kact = int(round(individual[idx + 2]))
        pt = int(round(individual[idx + 3]))
        ps = int(round(individual[idx + 4]))
        knum = max(KNUM_RANGE[0], min(KNUM_RANGE[1], knum))
        lnum = max(KSIZE_RANGE[0], min(KSIZE_RANGE[1], lnum))
        kact = max(KACT_RANGE[0], min(KACT_RANGE[1], kact))
        pt = max(PT_RANGE[0], min(PT_RANGE[1], pt))
        ps = max(PS_RANGE[0], min(PS_RANGE[1], ps))
        lstm_params.append([knum, lnum, kact, pt, ps])
        idx += 5
    
    bs = int(round(individual[idx]))
    opt = int(round(individual[idx + 1]))
    lr = individual[idx + 2]
    reg = int(round(individual[idx + 3]))
    bs = max(BS_RANGE[0], min(BS_RANGE[1], bs))
    opt = max(OPT_RANGE[0], min(OPT_RANGE[1], opt))
    lr = max(LR_RANGE[0], min(LR_RANGE[1], lr))
    reg = max(REG_RANGE[0], min(REG_RANGE[1], reg))
    setting = [bs, opt, lr, reg]
    
    return topo, cnn_params, lstm_params, setting


def get_individual_length():
    return INDIVIDUAL_LENGTH


def initialize_individual():
    n = NUM_MODULES
    bits_length = TOPO_BITS_LENGTH
    topo = [np.random.randint(0, 2) for _ in range(bits_length)]
    
    cnn_params = []
    for i in range(NUM_CNN_MODULES):
        knum = np.random.randint(KNUM_RANGE[0], KNUM_RANGE[1] + 1)
        ksize = np.random.randint(KSIZE_RANGE[0], KSIZE_RANGE[1] + 1)
        kact = np.random.randint(KACT_RANGE[0], KACT_RANGE[1] + 1)
        pt = np.random.randint(PT_RANGE[0], PT_RANGE[1] + 1)
        ps = np.random.randint(PS_RANGE[0], PS_RANGE[1] + 1)
        cnn_params.append([knum, ksize, kact, pt, ps])
    
    lstm_params = []
    for i in range(NUM_LSTM_MODULES):
        knum = np.random.randint(KNUM_RANGE[0], KNUM_RANGE[1] + 1)
        lnum = np.random.randint(KSIZE_RANGE[0], KSIZE_RANGE[1] + 1)
        kact = np.random.randint(KACT_RANGE[0], KACT_RANGE[1] + 1)
        pt = np.random.randint(PT_RANGE[0], PT_RANGE[1] + 1)
        ps = np.random.randint(PS_RANGE[0], PS_RANGE[1] + 1)
        lstm_params.append([knum, lnum, kact, pt, ps])
    
    bs = np.random.randint(BS_RANGE[0], BS_RANGE[1] + 1)
    opt = np.random.randint(OPT_RANGE[0], OPT_RANGE[1] + 1)
    lr = np.random.uniform(LR_RANGE[0], LR_RANGE[1])
    reg = np.random.randint(REG_RANGE[0], REG_RANGE[1] + 1)
    setting = [bs, opt, lr, reg]
    
    return encode_individual(topo, cnn_params, lstm_params, setting)


def is_valid_individual(individual):
    try:
        topo, cnn_params, lstm_params, setting = decode_individual(individual)
        n = NUM_MODULES
        for i in range(1, n):
            num_incoming = 0
            k = 0
            for ii in range(n):
                for jj in range(ii + 1, n):
                    if jj == i and topo[k] == 1:
                        num_incoming += 1
                    k += 1
            if num_incoming == 0:
                return False
        return True
    except:
        return False


def initialize_population(pop_size, train_dataset):
    population = []
    attempts = 0
    max_attempts = pop_size * 100
    
    while len(population) < pop_size and attempts < max_attempts:
        ind = initialize_individual()
        if is_valid_individual(ind):
            try:
                topo, cnn_params, lstm_params, setting = decode_individual(ind)
                batch_size, _, _, _ = decode_hyperparams(setting)
                if batch_size <= len(train_dataset):
                    population.append(ind)
            except:
                pass
        attempts += 1
    
    if len(population) < pop_size:
        raise ValueError(f"初始化失败:仅生成{len(population)}/{pop_size}个有效个体")
    
    return population


def decode_hyperparams(setting):
    bs, opt, lr, reg = setting
    batch_size = BATCH_SIZE_MAP[bs]
    learn_rate = lr
    optimizer_type = OPTIMIZER_MAP[opt]
    reg_type = REGULARIZER_MAP[reg]
    return batch_size, learn_rate, optimizer_type, reg_type