class Worker:
    def __init__(self, env, action_limits: tuple):
        self.env = env
        self.action = 0
        self.action_limits = action_limits

    def step(self, ds) -> float:
        yield self.action
        yield True

        while True:
            yield False

    def set_action(self, new_action):
        if new_action >= self.action_limits[1]:
            new_action = self.action_limits[1]

        if new_action <= self.action_limits[0]:
            new_action = self.action_limits[0]

        self.action = new_action

    def get_action(self):
        return self.action
