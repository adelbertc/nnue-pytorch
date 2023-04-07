import features
import nnue_dataset
import remote
import serialize

import argparse
import chess
import chess.pgn
import json
import os
import sys
import torch
from octoml_profile import accelerate, remote_profile, RemoteInferenceSession

FEATURE_SET = features.get_feature_set_from_name("HalfKAv2_hm")
MAX_PLY = 128
MATE = 99999
MIN_MATE_SCORE = MATE - MAX_PLY

# chess.PieceType are integers for PNBRQK, 1-7
# Create a "1-indexed" array with the piece value
PIECE_VALUES = [None, 100, 300, 300, 500, 900, 0]


# Copied and adapted from python-chess-engine-extensions
# https://github.com/Mk-Chan/python-chess-engine-extensions/blob/master/search/alphabeta.py
def sort_moves(board, move):
    attacking_piece = board.piece_at(move.from_square)
    attacked_piece = board.piece_at(move.to_square)

    order = 0
    if attacked_piece:
        order = PIECE_VALUES[attacked_piece.piece_type] - attacking_piece.piece_type
    return order


def next_fen(starting, move):
    board = chess.Board(starting)
    board.push(move)
    return board.fen()


# Negamax implementation from https://www.chessprogramming.org/Alpha-Beta
# with some help from python-chess-engine-extensions
# https://github.com/Mk-Chan/python-chess-engine-extensions/blob/master/search/alphabeta.py
#
# TODO
# * Add quiescence search https://www.chessprogramming.org/Quiescence_Search
#
def alpha_beta(fen, depth, alpha, beta, evaluate, ply=0):
    if depth == 0 or ply >= MAX_PLY:
        return evaluate(fen), []

    board = chess.Board(fen)

    if board.is_checkmate():
        mate_score = -MATE + ply
        return mate_score, []

    if (
        board.can_claim_draw()
        or board.is_insufficient_material()
        or board.is_stalemate()
    ):
        return 0, []

    legal_moves = sorted(
        board.legal_moves, reverse=True, key=lambda move: sort_moves(board, move)
    )

    best_eval = float("-inf")
    pv = []
    for move in legal_moves:
        new_fen = next_fen(fen, move)

        candidate_eval, candidate_pv = alpha_beta(
            new_fen, depth - 1, -beta, -alpha, evaluate, ply + 1
        )
        candidate_eval = -candidate_eval

        if candidate_eval >= beta:
            return beta, []

        if candidate_eval > best_eval:
            best_eval = candidate_eval

            if best_eval > alpha:
                alpha = best_eval
                pv = [move] + candidate_pv

    return alpha, pv


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


def eval_positions_with_search(model, fens, depth, inference_server=None):
    """
    Evaluate a batch of positions with the provided search depth.
    """
    results = []
    for fen in fens:
        score, pv = eval_position_with_search(model, fen, depth, inference_server)
        pv = get_algebraic(fen, pv)
        results.append((score, pv))
    return results


def eval_position_with_search(model, fen, depth, inference_server=None):
    """
    Evaluate a single position with the provided search depth.
    """

    if inference_server:
        evaluate = lambda fen: inference_server.evaluate(fen)
    else:
        evaluate = lambda fen: eval_positions(model, [fen])[0]

    return alpha_beta(
        fen,
        depth,
        float("-inf"),
        float("inf"),
        evaluate,
        True,
    )


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
    parser.add_argument("--fen", type=str, help="provide a single fen to evaluate")
    parser.add_argument("--fens", type=str, help="path to file of fens")
    parser.add_argument("--pgn", type=str, help="path to pgn")
    parser.add_argument("--depth", type=int, default=3, help="depth of search")
    parser.add_argument(
        "--remote", type=str, default=None, help="path to remote config file"
    )

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
                sys.exit(
                    "Profiling of evaluation with search is currently not supported."
                )
                # evaluations = eval_positions_with_search(model, fens, args.depth)
    else:
        if args.no_search:
            evaluations = eval_positions(model, fens)
        else:
            if args.remote:
                with open(args.remote, "r") as cf:
                    config = json.load(cf)
                inference_server = remote.RemoteInference(
                    config["endpoint"], config["token"]
                )
                print("Using remote inference server.")
            else:
                inference_server = None
                print("Using local inference.")
            evaluations = eval_positions_with_search(
                model, fens, args.depth, inference_server
            )

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
