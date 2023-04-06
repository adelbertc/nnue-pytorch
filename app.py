import features
import nnue_dataset
import serialize
import ui

import argparse
import chess
import chess.pgn
from python_chess_engine_extensions.evaluation.mixins import BaseEvaluation
from python_chess_engine_extensions.search.alphabeta import AlphaBetaMixin
import sys
import torch
from octoml_profile import accelerate, remote_profile, RemoteInferenceSession

FEATURE_SET = features.get_feature_set_from_name("HalfKAv2_hm")


class StockfishMixin(BaseEvaluation):
    def __init__(self, board, model):
        self.board = board
        self.model = model
        super().__init__(board)

    def evaluate(self):
        fen = self.board.fen()
        e = eval_positions(self.model, [fen])
        return e[0]


class StockfishEngine(AlphaBetaMixin, StockfishMixin):
    def __init__(self, board, model):
        AlphaBetaMixin.__init__(self, board)
        StockfishMixin.__init__(self, board, model)


def read_model(nnue_path):
    with open(nnue_path, "rb") as f:
        reader = serialize.NNUEReader(f, FEATURE_SET)
        return reader.model


def get_algebraic(starting_fen, moves):
    """
    Convert UCI moves to standard algebraic notation based
    on the provided FEN.
    """
    board = chess.Board(starting_fen)
    results = []
    for move in moves:
        an = board.san(move)
        if len(results) == 0:
            an = "...{}".format(an)

        results.append(an)
        board.push(move)
    return results


def eval_positions_with_search(model, fens, depth):
    """
    Evaluate a batch of positions with the provided search depth.
    """
    results = []
    for fen in fens:
        score, pv = eval_position_with_search(model, fen, depth)
        pv = get_algebraic(fen, pv)
        results.append((score, pv))
    return results


def eval_position_with_search(model, fen, depth):
    """
    Evaluate a single position with the provided search depth.
    """
    board = chess.Board(fen)
    engine = StockfishEngine(board, model)
    score, pv = engine.search(float("-inf"), float("inf"), depth)
    return score, pv


def eval_positions(model, fens):
    """
    Evaluate the list of positions with the model. There is no
    search tree depth involved in this evaluation, this function
    can drive a more sophisticated search-based evaluation function.

    Parameters
    ----------
    model
        The PyTorch model that evaluates the position
    fens
        List of FEN strings to evaluate
    """
    # Make a SparseBatch out of the FENs and extract the features
    batch = nnue_dataset.make_sparse_batch_from_fens(
        FEATURE_SET, fens, [0] * len(fens), [1] * len(fens), [0] * len(fens)
    )
    (
        us,
        them,
        white_indices,
        white_values,
        black_indices,
        black_values,
        outcome,
        score,
        psqt_indices,
        layer_stack_indices,
    ) = batch.contents.get_tensors("cuda")

    # Evaluate the positions and scale them to pawn scores
    evals = [
        v.item()
        for v in model.forward(
            us,
            them,
            white_indices,
            white_values,
            black_indices,
            black_values,
            psqt_indices,
            layer_stack_indices,
        )
        * 600.0
        / 208.0
    ]

    # Set the correct evaluation depending on perspective of score
    # (e.g. white has advantage vs. black has advantage)
    for i in range(len(evals)):
        if them[i] > 0.5:
            evals[i] = -evals[i]

    nnue_dataset.destroy_sparse_batch(batch)
    return evals


def filter_fens(fens):
    def not_in_check(fen):
        board = chess.Board(fen=fen)
        return not board.is_check()

    return list(filter(not_in_check, fens))


def pgn_to_fens(game):
    board = chess.Board()
    fens = []

    fens.append(board.fen())
    for move in game.mainline_moves():
        board.push(move)
        fens.append(board.fen())
    return fens


def main():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--net", type=str, help="path to a .nnue net")
    parser.add_argument(
        "--ui", action="store_true", help="run the UI to play against the computer"
    )
    parser.add_argument("--fen", type=str, help="provide a single fen to evaluate")
    parser.add_argument("--fens", type=str, help="path to file of fens")
    parser.add_argument("--pgn", type=str, help="path to pgn")
    parser.add_argument("--depth", type=int, default=3, help="depth of search")

    parser.add_argument(
        "--profile", action="store_true", help="run in PyTorch profiling mode"
    )
    parser.add_argument(
        "--no-search",
        action="store_true",
        help="run Stockfish evaluation without search",
    )
    args = parser.parse_args()

    model = read_model(args.net)
    model.eval()
    model.cuda()

    if args.profile:
        model = accelerate(model)

    if args.ui:
        next_move = lambda fen: eval_position_with_search(model, fen, args.depth)[1][0]
        ui.play_game(next_move)
    else:
        if args.pgn:
            with open(args.pgn) as pgnfile:
                game = chess.pgn.read_game(pgnfile)
            fens = pgn_to_fens(game)
        elif args.fen:
            fens = [args.fen]
        else:
            fens = open(args.fen).read().splitlines()

        fens = filter_fens(fens)

        board = chess.Board()
        engine = StockfishEngine(board, model)

        if args.profile:
            torch._dynamo.config.suppress_errors = True

            session = RemoteInferenceSession(
                [
                    "g4dn.xlarge/onnxrt-cuda",
                    "g4dn.xlarge/onnxrt-tensorrt",
                    "g4dn.xlarge/torch-eager-cuda",
                    "g4dn.xlarge/torch-inductor-cuda",
                ]
            )

            with remote_profile(session):
                if args.no_search:
                    for i in range(5):
                        evaluations = eval_positions(model, fens)
                else:
                    sys.exit("Profiling of evaluation with search is currently not supported.")
                    # evaluations = eval_positions_with_search(model, fens, args.depth)
        else:
            if args.no_search:
                evaluations = eval_positions(model, fens)
            else:
                evaluations = eval_positions_with_search(model, fens, args.depth)

        for i in range(len(fens)):
            fen = fens[i]
            if args.no_search:
                score = evaluations[i]
                print('[eval: {}], position = "{}"'.format(score, fen))
            else:
                score, moves = evaluations[i]
                moves_string = " ".join(moves)
                print('[eval: {}] {}, position = "{}"'.format(score, moves_string, fen))


if __name__ == "__main__":
    main()
