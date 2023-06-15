"""Example OctoAI service scaffold: Hello World."""
import app
import serialize

from octoai.service import Service
from octoai.types import Text

def read_model(nnue_path):
    with open(nnue_path, "rb") as f:
        reader = serialize.NNUEReader(f, app.FEATURE_SET)
        return reader.model

class Vida(Service):
    """An OctoAI service extends octoai.service.Service."""

    def setup(self):
        """Perform intialization."""
        model = app.read_model("data/nn-6877cd24400e.nnue")
        model.eval()
        model.cuda()
        self.model = model

    def infer(self, fen: Text) -> Text:
        """Perform inference."""
        evaluation, pv = app.eval_position_with_search(self.model, fen.text, depth=3)
        next_move_string = app.get_algebraic(fen.text, pv)[0]
        return Text(text=f"{evaluation} {next_move_string}")
