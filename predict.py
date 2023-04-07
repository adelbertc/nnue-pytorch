import app
import serialize

from cog import BasePredictor, Input, Path


def read_model(nnue_path):
    with open(nnue_path, "rb") as f:
        reader = serialize.NNUEReader(f, FEATURE_SET)
        return reader.model


class Predictor(BasePredictor):
    def setup(self):
        model = app.read_model("data/nn-6877cd24400e.nnue")
        model.eval()
        model.cuda()
        self.model = model

    def predict(
        self, fen: str = Input(description="FEN position to evaluate")
    ) -> float:
        return app.eval_positions(self.model, [fen])[0]
