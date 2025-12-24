class FeatureMapCollector:
    def __init__(self):
        self.maps = []

    def add(self, fmap):
        self.maps.append(fmap)

    def clear(self):
        self.maps = []

    def get(self):
        return self.maps
