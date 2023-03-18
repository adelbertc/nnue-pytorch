import features
import nnue_dataset
import serialize

import argparse
import chess
import torch
from octoml_profile import (accelerate, remote_profile, RemoteInferenceSession)

FEATURE_SET = features.get_feature_set_from_name("HalfKAv2_hm")

def read_model(nnue_path):
    with open(nnue_path, "rb") as f:
        reader = serialize.NNUEReader(f, FEATURE_SET)
        return reader.model

def eval_positions(model, fens, profile):
    """
    Evaluate the list of positions with the model.

    Parameters
    ----------
    model
        The PyTorch model that evaluates the position
    fens
        List of FEN strings to evaluate
    """
    if profile:
        model = accelerate(model)

    # Make a SparseBatch out of the FENs and extract the features
    batch = nnue_dataset.make_sparse_batch_from_fens(FEATURE_SET, fens, [0] * len(fens), [1] * len(fens), [0] * len(fens))
    us, them, white_indices, white_values, black_indices, black_values, outcome, score, psqt_indices, layer_stack_indices = batch.contents.get_tensors("cuda")

    # Evaluate the positions and scale them to pawn scores
    evals = [v.item() for v in model.forward(us, them, white_indices, white_values, black_indices, black_values, psqt_indices, layer_stack_indices) * 600.0 / 208.0]

    # Set the correct evaluation depending on perspective of score
    # (e.g. white has advantage vs. black has advantage)
    for i in range(len(evals)):
        if them[i] > 0.5:
            evals[i] = -evals[i]
        print("eval = {} for position = \"{}\"".format(evals[i], fens[i]))

    nnue_dataset.destroy_sparse_batch(batch)
    return evals

def filter_fens(fens):
    def not_in_check(fen):
        board = chess.Board(fen=fen)
        return not board.is_check()

    return list(filter(not_in_check, fens))

def main():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--net", type=str, help="path to a .nnue net")
    parser.add_argument("--fens", type=str, help="path to file of fens")
    parser.add_argument("--profile", action="store_true", help="run in PyTorch profiling mode")
    args = parser.parse_args()

    model = read_model(args.net)
    model.eval()
    model.cuda()
    
    fens = filter_fens(open(args.fens).read().splitlines())
   
    if args.profile:
        torch._dynamo.config.suppress_errors = True

        session = RemoteInferenceSession()
        with remote_profile(session):
            for i in range(10):
                eval_positions(model, fens, args.profile)
    else:
        eval_positions(model, fens, args.profile)

if __name__ == "__main__":
    main()
