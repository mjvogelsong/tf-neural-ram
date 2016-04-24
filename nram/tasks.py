import numpy as np

class Task(object):
    """
    Abstract Task
    """
    def generate_random_input(self, difficulty, memory_dim):
        """
        Generate a random values trace that is a valid
        input for the task.
        """
        raise NotImplementedError # abstract

    def run(self, values):
        """
        Run the task on the values trace,
        and return the result as a value trace.
        """
        raise NotImplementedError # abstract


class Access(Task):
    def generate_random_input(self, difficulty, memory_dim):
        """
        Input is given as k, A[0], ... , A[n-1], 0.0, ...
        """
        size = memory_dim - 2 # np.random.randint(1, memory_dim-1)
        k = np.random.randint(0, size)
        random_input = np.random.randint(1, memory_dim, size=memory_dim)
        random_input[0] = k
        random_input[size+1] = 0
        return random_input

    def run(self, values):
        """
        Given a value k and an array A, return A[k].
        Input is given as k, A[0], ... , A[size-1], 0.0, ...
        and the network should replace the first
        memory cell with A[k].
        """
        values_copy = list(values)
        k = values_copy[0]
        values_copy[0] = values_copy[k+1]
        return values_copy
