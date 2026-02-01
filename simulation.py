class SIMULATION:
    """
    Transitional SIMULATION class:
    - For now, delegates to simulate_legacy.main() so behavior stays identical.
    - Next gates will move logic into WORLD/ROBOT/SENSOR/MOTOR and remove this bridge.
    """
    def __init__(self):
        pass

    def Run(self):
        import simulate_legacy as legacy
        legacy.main()
