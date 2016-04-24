import numpy as np

def values_to_distribution_memory(values):
    """
    Transform a list of values into a 2D numpy array (aka memory)
    of probaility distributions.

    Int -> 1.0 at int index, 0.0 everywhere else
    None -> Uniformly distributed 1.0/len(values)

    E.g. [2, 1, None]
    ->
    nparray
    [
    [0.0, 0.0, 1.0],
    [0.0, 1.0, 0.0],
    [0.333, 0.333, 0.333]
    ]
    """
    length = len(values)
    memory = np.zeros([length, length], dtype=np.float32)
    uniform_density = 1.0 / float(length)
    for i, val in enumerate(values):
        if val != None:
            memory[i][val] = 1.0
        else:
            memory[i].fill(uniform_density)
    return memory

def batch_to_distribution_memories(batch):
    """
    Transform a batch of value lists into a batch
    of 2D numpy arrays of probability distributions.
    """
    batch_memory = []
    for value_array in batch:
        batch_memory.append(values_to_distribution_memory(value_array))
    return batch_memory


def next_batch(memory_dim, batch_length, task):
    """
    Generate a batch of input/target memory pairs,
    where the task specifies the target transformation.
    """
    batch_inputs = []
    batch_targets = []
    difficulty = 0.0
    for _ in xrange(batch_length):
        # Values as integers
        random_input = task.generate_random_input(difficulty, memory_dim)
        target_output = task.run(random_input)

        # Values as MxM memories
        batch_inputs.append(values_to_distribution_memory(random_input))
        batch_targets.append(values_to_distribution_memory(target_output))

    return batch_inputs, batch_targets
