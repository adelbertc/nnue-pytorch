import app
import serialize

from cog import BaseModel, BasePredictor, Input

class Output(BaseModel):
    evaluation: float
    next_move: str


def read_model(nnue_path):
    with open(nnue_path, "rb") as f:
        reader = serialize.NNUEReader(f, app.FEATURE_SET)
        return reader.model


class Predictor(BasePredictor):
    def setup(self):
        model = app.read_model("data/nn-6877cd24400e.nnue")
        model.eval()
        model.cuda()
        self.model = model

    def predict(
        self,
        fen: str = Input(description="FEN position to evaluate"),
        depth: int = Input(description="Depth to search in the tree", default=1, ge=0)
    ) -> Output:
        evaluation, pv = app.eval_position_with_search(self.model, fen, depth)
        next_move_string = app.get_algebraic(fen, pv)[0]
        return Output(evaluation=evaluation, next_move=next_move_string)
