from ar_shuffleboard.controllers.main_controller import MainController

from ar_shuffleboard.utils.config_loader import load_config


def main():
    print("Hello from seoultech-computervision-ar-shuffleboard!")

    config = load_config()

    controller = MainController(**config)
    controller.run()


if __name__ == "__main__":
    main()
