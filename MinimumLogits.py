import torch
import Tournament

# t = Tournament.Tournament(100)
t = Tournament.Tournament(10)

# x = torch.ones(99*50) * .5
x = torch.ones(9*5) * .5
# x[:9] = 0
x[:9] = 0

y = t(x)

print(y)