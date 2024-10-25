class EarlyStopping:
    def __init__(self, stopping_delta = 1e-6):
        self.delta_label = None
        self.stopping_delta = stopping_delta
        self.is_early_stop = False

    def __call__(self, predicted, predicted_previous):
        delta_label = (
            float((predicted != predicted_previous).float().sum().item())
            / predicted_previous.shape[0]
        )
        if self.stopping_delta > delta_label:
            self.is_early_stop = True
        self.delta_label = delta_label