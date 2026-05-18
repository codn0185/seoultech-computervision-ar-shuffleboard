from ar_shuffleboard.models.main_model import MainModel
from ar_shuffleboard.views.main_view import MainView


class MainController:
    def __init__(self):
        self.model = MainModel(self)
        self.view = MainView(self)

    def run(self):
        pass
