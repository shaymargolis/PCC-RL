
class DS:
    def __init__(self, obs, reward):
        self.data = (obs, reward)


class WorkerRunner():
    def __init__(self, workers: list, obs_init, reward_init):
        self.workers = workers

        self.ds = [DS(obs_init, reward_init), DS(obs_init, reward_init)]
        self.gen = [self.workers[i].step(self.ds[i]) for i in range(len(self.workers))]

    def get_next_rate(self, ind):
        try:
            n = next(self.gen[ind])
        except StopIteration as ex:
            self.gen[ind] = self.workers[ind].step(self.ds[ind])
            return self.get_next_rate(ind)

        return n

    def finish_substep(self, ind, obs, reward):
        self.ds[ind].data = (obs, reward)

        try:
            next(self.gen[ind])
        except StopIteration as ex:
            return False

        return True

    def start_step(self):
        return [self.get_next_rate(i) for i in range(len(self.workers))]

    def finish_step(self, obs, rewards):
        for i in range(len(self.workers)):
            self.finish_substep(i, obs[i], rewards[i])
