
class NoRegretCombining:
    def __init__(self):
        pass

    def predict(self, observation: float):
        #  Update action according to observation
        gradient = self.last_direction_choice * observation / self.delta
        self.action = self.project_action(self.action - self.mu * gradient)

        self.update_gradient_ascent_speed()

        #  Generate next random choice
        direction = self.get_direction_randomly()
        return self.action + direction * self.delta
