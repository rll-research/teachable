import random

class Buffer:
    def __init__(self, buffer_capacity, separate_by_level):
        self.buffer_capacity = buffer_capacity
        self.separate_by_level = separate_by_level
        if separate_by_level:
            self.buffer = [[]]
        else:
            self.buffer = []

    def add_batch(self, batch, level):
        if self.separate_by_level:
            if level >= len(self.buffer):
                self.buffer.append([batch])
            else:
                level_buffer = self.buffer[level]
                if len(level_buffer) < self.buffer_capacity:
                    level_buffer.append(batch)
                else:
                    index = random.randint(0, self.buffer_capacity - 1)
                    level_buffer[index] = batch

        else:
            if len(self.buffer) < self.buffer_capacity:
                self.buffer.append(batch)
            else:
                index = random.randint(0, self.buffer_capacity - 1)
                self.buffer[index] = batch

    def sample(self):
        if self.separate_by_level:
            level_buffer = random.choice(self.buffer)
            return random.choice(level_buffer)
        else:
            return random.choice(self.buffer)



# TODO:
#  Lots of things we could consider!
#  (1) sequential eviction
#  (2) store as arrays not batches
#  (3) preferentially sample from the current level
#  (4) Add flags for buffer