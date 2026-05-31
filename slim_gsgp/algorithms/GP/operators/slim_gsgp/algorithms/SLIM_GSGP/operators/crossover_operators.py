"""
Crossover operators for SLIM-GSGP as introduced in:
"Introducing Crossover in SLIM-GSGP" (EuroGP 2025)
"""

import random
import torch
import copy

def swap_crossover(parent1, parent2):
    """
    Swap Crossover (XOSw) for SLIM-GSGP.
    """
    blocks1 = parent1.collection
    sem1 = parent1.train_semantics
    blocks2 = parent2.collection
    sem2 = parent2.train_semantics

    len1 = len(blocks1)
    len2 = len(blocks2)
    min_len = min(len1, len2)
    
    # Generate random mask
    mask = [random.choice([0, 1]) for _ in range(min_len)]

    # Build first child
    child1_blocks = []
    child1_sem_rows = []
    for i in range(min_len):
        if mask[i] == 0:
            child1_blocks.append(blocks1[i])
            child1_sem_rows.append(sem1[i].unsqueeze(0))
        else:
            child1_blocks.append(blocks2[i])
            child1_sem_rows.append(sem2[i].unsqueeze(0))
    
    # Add remaining blocks from the longer parent
    if len1 > min_len:
        for i in range(min_len, len1):
            child1_blocks.append(blocks1[i])
            child1_sem_rows.append(sem1[i].unsqueeze(0))
    else:
        for i in range(min_len, len2):
            child1_blocks.append(blocks2[i])
            child1_sem_rows.append(sem2[i].unsqueeze(0))
    
    child1_sem = torch.cat(child1_sem_rows, dim=0) if child1_sem_rows else torch.empty(0)

    # Build second child (inverse mask)
    child2_blocks = []
    child2_sem_rows = []
    for i in range(min_len):
        if mask[i] == 0:
            child2_blocks.append(blocks2[i])
            child2_sem_rows.append(sem2[i].unsqueeze(0))
        else:
            child2_blocks.append(blocks1[i])
            child2_sem_rows.append(sem1[i].unsqueeze(0))
    
    if len1 > min_len:
        for i in range(min_len, len2):
            child2_blocks.append(blocks2[i])
            child2_sem_rows.append(sem2[i].unsqueeze(0))
    else:
        for i in range(min_len, len1):
            child2_blocks.append(blocks1[i])
            child2_sem_rows.append(sem1[i].unsqueeze(0))
    
    child2_sem = torch.cat(child2_sem_rows, dim=0) if child2_sem_rows else torch.empty(0)

    child1 = create_individual(parent1, child1_blocks, child1_sem)
    child2 = create_individual(parent2, child2_blocks, child2_sem)
    return child1, child2

def donor_crossover(parent_donor, parent_receiver):
    """
    Donor Crossover (XODn) for SLIM-GSGP.
    """
    donor_blocks = parent_donor.collection
    donor_sem = parent_donor.train_semantics
    receiver_blocks = parent_receiver.collection
    receiver_sem = parent_receiver.train_semantics

    if not donor_blocks:
        return parent_donor, parent_receiver

    idx = random.randrange(len(donor_blocks))
    selected_block = donor_blocks[idx]
    selected_sem = donor_sem[idx].unsqueeze(0)

    # Child 1: donor without the selected block
    child1_blocks = donor_blocks[:idx] + donor_blocks[idx+1:]
    if donor_sem.size(0) > 1:
        child1_sem = torch.cat([donor_sem[:idx], donor_sem[idx+1:]], dim=0)
    else:
        child1_sem = torch.empty(0)

    # Child 2: receiver with the new block added
    child2_blocks = receiver_blocks + [selected_block]
    child2_sem = torch.cat([receiver_sem, selected_sem], dim=0)

    child1 = create_individual(parent_donor, child1_blocks, child1_sem)
    child2 = create_individual(parent_receiver, child2_blocks, child2_sem)
    return child1, child2

def create_individual(base_individual, new_blocks, new_train_semantics):
    """
    Helper function to create a new individual from blocks and semantics.
    """
    new_ind = copy.deepcopy(base_individual)
    new_ind.collection = new_blocks
    new_ind.train_semantics = new_train_semantics
    if hasattr(new_ind, 'test_semantics'):
        new_ind.test_semantics = None
    if hasattr(new_ind, 'fitness'):
        new_ind.fitness = None
    return new_ind
