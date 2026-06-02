class BaseGenEvalImpl:
    def __init__(self, dataset, pipeline, args):
        self.config = args
        self.dataset = dataset
        self.pipeline = pipeline

    def analyze_result(self):
        raise NotImplementedError("analyze_result() in BaseGenEvalImpl is required.")

    def __call__(self):
        raise NotImplementedError("__call__() in BaseGenEvalImpl is required.")
