from src.gym.worker.worker import Worker


class RewardCalculator:
    def __call__(self, ds, worker: Worker):
        """
        Recieves a DataSource object for forwarding the reward
        to the worker, and the worker to run a step on and calculate
        the reward for.
        :param ds: The dataSource
        :param worker: The worker to run
        :return: The reward of the worker from the step
        """
        return 0


class AverageRewardCalculator(RewardCalculator):
    def __call__(self, ds, worker: Worker):
        gen = worker.step(ds)

        reward = 0
        reward_steps = 0

        while True:
            try:
                action = next(gen)
            except StopIteration as ex:
                break

            yield action
            yield True

            reward += ds.data[1]
            reward_steps += 1

            #  Continue to next action
            next(gen)

        if reward_steps == 0:
            print("[ERROR] Worker's number of steps must be atleast 1.")
            return None

        return reward / reward_steps


class LastOccurrenceRewardCalculator(RewardCalculator):
    def __call__(self, ds, worker: Worker) -> float:
        gen = worker.step(ds)

        reward = 0
        reward_steps = 0

        while True:
            try:
                action = next(gen)
            except StopIteration as ex:
                break

            yield action
            yield True

            reward = ds.data[1]
            reward_steps += 1

            #  Continue to next action
            next(gen)

        if reward_steps == 0:
            print("[ERROR] Worker's number of steps must be atleast 1.")
            return None

        return reward
