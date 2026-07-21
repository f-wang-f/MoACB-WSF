import numpy as np

from .config import (
    NUM_MODULES, TOPO_BITS_LENGTH, INDIVIDUAL_LENGTH, CROSSOVER_PROB,
    KNUM_RANGE, KSIZE_RANGE, KACT_RANGE, PT_RANGE, PS_RANGE,
    BS_RANGE, OPT_RANGE, LR_RANGE, REG_RANGE
)
from .encoding import encode_individual, decode_individual, is_valid_individual


def fast_non_dominated_sort(performance, complexity):
    pop_size = len(performance)
    fronts = []
    rank = np.zeros(pop_size, dtype=int)
    domination_count = np.zeros(pop_size, dtype=int)
    dominated_solutions = [[] for _ in range(pop_size)]
    valid_mask = np.isfinite(performance) & np.isfinite(complexity)
    
    for i in range(pop_size):
        if not valid_mask[i]:
            continue
        for j in range(pop_size):
            if i == j or not valid_mask[j]:
                continue
            if (performance[i] <= performance[j] and complexity[i] <= complexity[j]) and \
                    (performance[i] < performance[j] or complexity[i] < complexity[j]):
                dominated_solutions[i].append(j)
            elif (performance[j] <= performance[i] and complexity[j] <= complexity[i]) and \
                    (performance[j] < performance[i] or complexity[j] < complexity[i]):
                domination_count[i] += 1
    
    current_front = np.where((domination_count == 0) & valid_mask)[0]
    if len(current_front) == 0:
        valid_indices = np.where(valid_mask)[0]
        current_front = valid_indices[:1] if len(valid_indices) > 0 else np.array([0])
    
    fronts.append(current_front)
    rank[current_front] = 0
    front_idx = 1
    
    while len(fronts[-1]) > 0:
        next_front = []
        for i in fronts[-1]:
            for j in dominated_solutions[i]:
                domination_count[j] -= 1
                if domination_count[j] == 0:
                    next_front.append(j)
                    rank[j] = front_idx
        
        if next_front:
            fronts.append(np.array(next_front))
        else:
            break
        front_idx += 1
    
    rank[~valid_mask] = front_idx + 1
    return fronts, rank


def crowding_distance(performance, complexity, fronts):
    pop_size = len(performance)
    distance = np.zeros(pop_size)
    
    for front in fronts:
        if len(front) <= 2:
            distance[front] = np.inf if len(front) == 2 else 0
            continue
        
        valid_mask = np.isfinite(performance[front]) & np.isfinite(complexity[front])
        if not valid_mask.any():
            continue
        
        front_valid = front[valid_mask]
        sorted_perf_idx = np.argsort(performance[front_valid])
        distance[front_valid[sorted_perf_idx[0]]] = np.inf
        distance[front_valid[sorted_perf_idx[-1]]] = np.inf
        
        sorted_comp_idx = np.argsort(complexity[front_valid])
        distance[front_valid[sorted_comp_idx[0]]] = np.inf
        distance[front_valid[sorted_comp_idx[-1]]] = np.inf
        
        perf_range = performance[front_valid[sorted_perf_idx[-1]]] - performance[front_valid[sorted_perf_idx[0]]]
        comp_range = complexity[front_valid[sorted_comp_idx[-1]]] - complexity[front_valid[sorted_comp_idx[0]]]
        
        if perf_range > 0:
            for i in range(1, len(front_valid) - 1):
                idx = sorted_perf_idx[i]
                prev_idx = sorted_perf_idx[i - 1]
                next_idx = sorted_perf_idx[i + 1]
                distance[front_valid[idx]] += (performance[front_valid[next_idx]] - performance[
                    front_valid[prev_idx]]) / perf_range
        
        if comp_range > 0:
            for i in range(1, len(front_valid) - 1):
                idx = sorted_comp_idx[i]
                prev_idx = sorted_comp_idx[i - 1]
                next_idx = sorted_comp_idx[i + 1]
                distance[front_valid[idx]] += (complexity[front_valid[next_idx]] - complexity[
                    front_valid[prev_idx]]) / comp_range
    
    return distance


def tournament_selection(population, rank, distance, pop_size):
    mating_pool = []
    pop_size_actual = len(population)
    
    for _ in range(pop_size):
        idx1, idx2 = np.random.randint(0, pop_size_actual, 2)
        
        if not np.isfinite(rank[idx1]):
            rank[idx1] = 1e6
        if not np.isfinite(rank[idx2]):
            rank[idx2] = 1e6
        if not np.isfinite(distance[idx1]):
            distance[idx1] = 0
        if not np.isfinite(distance[idx2]):
            distance[idx2] = 0
        
        if rank[idx1] < rank[idx2] or (rank[idx1] == rank[idx2] and distance[idx1] > distance[idx2]):
            mating_pool.append(population[idx1].copy())
        else:
            mating_pool.append(population[idx2].copy())
    
    return mating_pool


def environmental_selection(combined_pop, combined_perf, combined_complex, combined_rank, combined_dist, pop_size):
    sorted_idx = np.argsort(combined_rank)
    combined_pop = [combined_pop[i] for i in sorted_idx]
    combined_perf = combined_perf[sorted_idx]
    combined_complex = combined_complex[sorted_idx]
    combined_rank = combined_rank[sorted_idx]
    combined_dist = combined_dist[sorted_idx]
    
    new_pop = []
    new_perf = []
    new_complex = []
    current_size = 0
    
    for rank_val in np.unique(combined_rank):
        if current_size >= pop_size:
            break
        
        mask = combined_rank == rank_val
        front_pop = [combined_pop[i] for i in range(len(combined_pop)) if mask[i]]
        front_perf = combined_perf[mask]
        front_complex = combined_complex[mask]
        front_dist = combined_dist[mask]
        
        if current_size + len(front_pop) <= pop_size:
            new_pop.extend(front_pop)
            new_perf.extend(front_perf)
            new_complex.extend(front_complex)
            current_size += len(front_pop)
        else:
            remaining = pop_size - current_size
            valid_dist = np.isfinite(front_dist)
            
            if not valid_dist.any():
                selected = np.random.choice(len(front_pop), remaining, replace=False)
            else:
                sorted_dist_idx = np.argsort(front_dist[valid_dist])[::-1]
                selected = np.where(valid_dist)[0][sorted_dist_idx[:remaining]]
            
            new_pop.extend([front_pop[i] for i in selected])
            new_perf.extend(front_perf[selected])
            new_complex.extend(front_complex[selected])
            current_size = pop_size
    
    return new_pop, np.array(new_perf), np.array(new_complex)


def generate_valid_topology():
    n = NUM_MODULES
    bits_length = TOPO_BITS_LENGTH
    
    while True:
        topo = [np.random.randint(0, 2) for _ in range(bits_length)]
        valid = True
        
        for i in range(1, n):
            num_incoming = 0
            k = 0
            for ii in range(n):
                for jj in range(ii + 1, n):
                    if jj == i and topo[k] == 1:
                        num_incoming += 1
                    k += 1
            if num_incoming == 0:
                valid = False
                break
        
        if valid:
            return topo


def crossover_population(mating_pool):
    offspring = []
    pop_size = len(mating_pool)
    L = INDIVIDUAL_LENGTH
    L_t = TOPO_BITS_LENGTH
    
    for i in range(0, pop_size, 2):
        if i + 1 < pop_size:
            par1 = mating_pool[i]
            par2 = mating_pool[i + 1]
            
            if np.random.random() < CROSSOVER_PROB:
                k = np.random.randint(0, L)
                
                if k < L_t:
                    child1 = par1.copy()
                    child2 = par2.copy()
                    
                    child1[:L_t] = generate_valid_topology()
                    child2[:L_t] = generate_valid_topology()
                    
                    offspring.append(child1)
                    offspring.append(child2)
                else:
                    child1 = np.concatenate([par1[:k], par2[k:]])
                    child2 = np.concatenate([par2[:k], par1[k:]])
                    
                    offspring.append(child1)
                    offspring.append(child2)
            else:
                offspring.append(par1.copy())
                offspring.append(par2.copy())
        else:
            offspring.append(mating_pool[i].copy())
    
    return offspring


def variable_length_mutation(individual):
    topo, cnn_params, lstm_params, setting = decode_individual(individual)
    n = NUM_MODULES
    max_attempts = 100
    attempts = 0
    
    cnn_param_bounds = [KNUM_RANGE, KSIZE_RANGE, KACT_RANGE, PT_RANGE, PS_RANGE]
    lstm_param_bounds = [KNUM_RANGE, KSIZE_RANGE, KACT_RANGE, PT_RANGE, PS_RANGE]
    setting_bounds = [BS_RANGE, OPT_RANGE, LR_RANGE, REG_RANGE]
    
    while attempts < max_attempts:
        mutation_region = np.random.choice([0, 1, 2])
        new_topo = topo.copy()
        new_cnn_params = [p.copy() for p in cnn_params]
        new_lstm_params = [p.copy() for p in lstm_params]
        new_setting = setting.copy()
        mutated = False
        
        if mutation_region == 0:
            bits_length = len(topo)
            while True:
                candidate_topo = [np.random.randint(0, 2) for _ in range(bits_length)]
                if candidate_topo != topo:
                    temp_ind = encode_individual(candidate_topo, new_cnn_params, new_lstm_params, new_setting)
                    if is_valid_individual(temp_ind):
                        new_topo = candidate_topo
                        mutated = True
                        break
        
        elif mutation_region == 1:
            module_type = np.random.choice([0, 1])
            if module_type == 0:
                module_idx = np.random.randint(0, 3)
                param_idx = np.random.randint(0, 5)
                original_val = new_cnn_params[module_idx][param_idx]
                min_val, max_val = cnn_param_bounds[param_idx]
                candidate_vals = [v for v in range(min_val, max_val + 1) if v != original_val]
                if candidate_vals:
                    new_val = np.random.choice(candidate_vals)
                    new_cnn_params[module_idx][param_idx] = new_val
                    mutated = True
            else:
                module_idx = np.random.randint(0, 2)
                param_idx = np.random.randint(0, 5)
                original_val = new_lstm_params[module_idx][param_idx]
                min_val, max_val = lstm_param_bounds[param_idx]
                candidate_vals = [v for v in range(min_val, max_val + 1) if v != original_val]
                if candidate_vals:
                    new_val = np.random.choice(candidate_vals)
                    new_lstm_params[module_idx][param_idx] = new_val
                    mutated = True
        
        elif mutation_region == 2:
            param_idx = np.random.randint(0, 4)
            original_val = new_setting[param_idx]
            if param_idx == 2:
                min_val, max_val = setting_bounds[param_idx]
                while True:
                    new_val = np.random.uniform(min_val, max_val)
                    if abs(new_val - original_val) > 0.00001:
                        new_setting[param_idx] = new_val
                        mutated = True
                        break
            else:
                min_val, max_val = setting_bounds[param_idx]
                candidate_vals = [v for v in range(int(min_val), int(max_val) + 1) if v != original_val]
                if candidate_vals:
                    new_val = np.random.choice(candidate_vals)
                    new_setting[param_idx] = new_val
                    mutated = True
        
        if mutated:
            new_individual = encode_individual(new_topo, new_cnn_params, new_lstm_params, new_setting)
            if not np.array_equal(new_individual, individual) and is_valid_individual(new_individual):
                return new_individual
        
        attempts += 1
    
    return individual.copy()