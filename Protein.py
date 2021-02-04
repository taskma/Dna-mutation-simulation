class Protein():
    def __init__(self, inpName=None, inpSequence=None):
        self.count = 1
        if inpName != None:
            self.name = inpName
        if inpSequence != None:
            self.sequence = inpSequence