# class DataWrapper():
#     time = []
#     # strain = []
#     E = 0.0
#     etha = 0.0
#     # c01 = 0.0
#     # c10 =0.0
#     #
#
#     def __init__(self, time, E, etha):
#         self.time = time
#         # self.strain = strain
#         self.E = E
#         self.etha = etha
#
#         return None


class DataWrapper:
    E = 0.0
    etha = 0.0

    def __init__(self, data_type, data, stretch, E, etha, eps=None):
        self.FitStrainEnergy = False
        if data_type == "time":
            self.time = data
            self.strain = eps
        elif data_type == "strain":
            self.strain = data
            self.time = None

        self.stretch = stretch
        self.E = E
        self.etha = etha
