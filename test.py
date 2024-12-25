class ReturnNodeValue:
    def __init__(self, node_secret: str):
        self._value = node_secret

    def __call__(self):
        print(self._value)
    

node = ReturnNodeValue("I'm A")
node()