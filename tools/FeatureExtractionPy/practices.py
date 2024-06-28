class car:
    def __init__(self, speed = 40):
        self._speed = speed

    @property
    def speed(self):
        return self._speed

    @speed.setter
    def speed(self, speed):
        self._speed = speed
        return

c1 = car()

print(c1.speed)
c1.speed = 100
print(c1.speed)


