from cem import CEM

def func(x):
  _sum = 0
  for i, c in enumerate(x):
    tmp = c - int(i/2)
    _sum += tmp*tmp
  return _sum

if __name__ == '__main__':
  cem = CEM(func, 20)
  v = cem.eval()
  print(v, func(v))

