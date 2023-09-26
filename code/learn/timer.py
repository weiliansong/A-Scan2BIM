import time

class Timer:

  def __init__(self, name):
    self.name = name
    self._start = None
    self._end = None


  def __enter__(self):
    self._start = time.time()


  def __exit__(self, *exc_info):
    self._end = time.time()

    print('%s: %f' % (self.name, self._end - self._start))