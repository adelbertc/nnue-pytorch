import features
import nnue_dataset
import serialize

import argparse
import chess
import torch
from octoml_profile import (accelerate, remote_profile, RemoteInferenceSession)

torch._dynamo.config.suppress_errors = True

def read_model(nnue_path, feature_set):
    with open(nnue_path, "rb") as f:
        reader = serialize.NNUEReader(f, feature_set)
        return reader.model

def eval_position(model, fen, feature_set, dynamite):
    fens = [fen]
    batch = nnue_dataset.make_sparse_batch_from_fens(feature_set, fens, [0] * len(fens), [1] * len(fens), [0] * len(fens))

    us, them, white_indices, white_values, black_indices, black_values, outcome, score, psqt_indices, layer_stack_indices = batch.contents.get_tensors("cuda")
    evaluation = predict(model, us, them, white_indices, white_values, black_indices, black_values, outcome, score, psqt_indices, layer_stack_indices, dynamite)

    nnue_dataset.destroy_sparse_batch(batch)
    return evaluation

def predict(model, us, them, white_indices, white_values, black_indices, black_values, outcome, score, psqt_indices, layer_stack_indices, dynamite):
    if dynamite:
        model = accelerate(model)

    result = model.forward(us, them, white_indices, white_values, black_indices, black_values, psqt_indices, layer_stack_indices) * 600.0
    result = result.item() / 208.0

    if them[0] > 0.5:
       result = -result

    return result

def filter_fens(fens):
    def not_in_check(fen):
        board = chess.Board(fen=fen)
        return not board.is_check()
    return [fen for fen in fens if not_in_check(fen)]

def eval_positions(model, fens, feature_set, dynamite):
    for fen in fens:
        e = eval_position(model, fen, feature_set, dynamite)
        print("eval = {} for position = \"{}\"".format(e, fen))

def main():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--net", type=str, help="path to a .nnue net")
    parser.add_argument("--fens", type=str, help="path to file of fens")
    parser.add_argument("--dynamite", action="store_true", help="benchmark with dynamite")
    features.add_argparse_args(parser)
    args = parser.parse_args()

    feature_set = features.get_feature_set_from_name(args.features)
    model = read_model(args.net, feature_set)
    model.eval()
    model.cuda()
    
    fens = filter_fens(open(args.fens).read().splitlines())
   
    if args.dynamite:
        session = RemoteInferenceSession()
        with remote_profile(session):
            for i in range(10):
                eval_positions(model, fens, feature_set, args.dynamite)
    else:
        eval_positions(model, fens, feature_set, args.dynamite)

if __name__ == "__main__":
    main()
