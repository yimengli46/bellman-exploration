from random import Random

class RandomIterator:
    """Class to implement an iterator
    of powers of two"""

    def __init__(self, num_elems=0, seed=0):
        self.num_elems = num_elems
        self.random = Random(seed)

    def __iter__(self):
        self.n = 0
        return self

    def __next__(self):
        if self.n < self.num_elems:
            result = self.random.randint(0, 10) 
            self.n += 1
            return result
        else:
            raise StopIteration

num_elems = 100
seed = 3
# create an object
numbers = RandomIterator(num_elems, seed=seed)

# create an iterable from the object
my_iter = iter(numbers)

for i in range(num_elems):
	print(f'i = {i}, next = {next(my_iter)}')