from Incept.utils import seed_everything

class ExpManager:
    def __init__(self, config):
        self.config = config
    
    def run(self):
        seed_everything()
