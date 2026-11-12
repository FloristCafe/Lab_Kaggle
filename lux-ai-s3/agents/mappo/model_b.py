"""B. Persistent spatial evidence with bounded, differentiable match recurrence."""
import torch
from torch import nn

from .shared.model import SpatialActor, WIDTH


class Actor(SpatialActor):
    uses_memory = True

    def __init__(self):
        super().__init__()
        self.episodes = nn.GRU(4, 16, batch_first=True)
        self.memory_key = nn.Linear(16, WIDTH)
        self.memory_read = nn.Linear(WIDTH * 2, WIDTH)
        self.unit_read = nn.Linear(WIDTH, WIDTH)
        nn.init.zeros_(self.memory_read.weight)
        nn.init.constant_(self.memory_read.bias, -2)

    def contextualize(self, tiles, tokens, state):
        batch = tiles.shape[0]
        memory = state["memory"].float().flatten(3).permute(0, 3, 1, 2)
        memory = memory.reshape(batch * 576, 5, 4)
        sequence, _ = self.episodes(memory)
        sequence = sequence.reshape(batch, 576, 5, 16)
        count = (state["memory_count"] - 1).view(batch, 1, 1, 1).expand(-1, 576, 1, 16)
        read = self.memory_key(sequence.gather(2, count).squeeze(2))
        gate = torch.sigmoid(self.memory_read(torch.cat((tiles, read), dim=-1)))
        read = gate * read
        return tiles + read, tokens + self.unit_read(read.mean(1, keepdim=True))
