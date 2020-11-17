from src.gym.worker.worker import Worker


class ConstantWorker(Worker):
    def __init__(self, env, action_limits, constant_speed):
        super().__init__(env, action_limits)

        self.constant_speed = constant_speed

    def set_constant_speed(self, new):
        self.constant_speed = new

    def step(self, ds) -> float:
        self.set_action(self.constant_speed)

        yield self.constant_speed
        yield True
