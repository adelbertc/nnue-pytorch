"""
Copied and adapted from the chess demo code for PySimpleGUI.

https://github.com/PySimpleGUI/PySimpleGUI
"""

import argparse
from chess import Board, Move, WHITE
import copy
import os
import PySimpleGUI as sg
import sys

CHESS_PATH = "data"  # path to the chess pieces

BLANK = 0  # piece names
PAWNB = 1
KNIGHTB = 2
BISHOPB = 3
ROOKB = 4
KINGB = 5
QUEENB = 6
PAWNW = 7
KNIGHTW = 8
BISHOPW = 9
ROOKW = 10
KINGW = 11
QUEENW = 12

initial_board = [
    [ROOKB, KNIGHTB, BISHOPB, QUEENB, KINGB, BISHOPB, KNIGHTB, ROOKB],
    [
        PAWNB,
    ]
    * 8,
    [
        BLANK,
    ]
    * 8,
    [
        BLANK,
    ]
    * 8,
    [
        BLANK,
    ]
    * 8,
    [
        BLANK,
    ]
    * 8,
    [
        PAWNW,
    ]
    * 8,
    [ROOKW, KNIGHTW, BISHOPW, QUEENW, KINGW, BISHOPW, KNIGHTW, ROOKW],
]

blank = os.path.join(CHESS_PATH, "blank.png")
bishopB = os.path.join(CHESS_PATH, "nbishopb.png")
bishopW = os.path.join(CHESS_PATH, "nbishopw.png")
pawnB = os.path.join(CHESS_PATH, "npawnb.png")
pawnW = os.path.join(CHESS_PATH, "npawnw.png")
knightB = os.path.join(CHESS_PATH, "nknightb.png")
knightW = os.path.join(CHESS_PATH, "nknightw.png")
rookB = os.path.join(CHESS_PATH, "nrookb.png")
rookW = os.path.join(CHESS_PATH, "nrookw.png")
queenB = os.path.join(CHESS_PATH, "nqueenb.png")
queenW = os.path.join(CHESS_PATH, "nqueenw.png")
kingB = os.path.join(CHESS_PATH, "nkingb.png")
kingW = os.path.join(CHESS_PATH, "nkingw.png")

images = {
    BISHOPB: bishopB,
    BISHOPW: bishopW,
    PAWNB: pawnB,
    PAWNW: pawnW,
    KNIGHTB: knightB,
    KNIGHTW: knightW,
    ROOKB: rookB,
    ROOKW: rookW,
    KINGB: kingB,
    KINGW: kingW,
    QUEENB: queenB,
    QUEENW: queenW,
    BLANK: blank,
}


def render_square(image, key, location):
    if (location[0] + location[1]) % 2:
        color = "#B58863"
    else:
        color = "#F0D9B5"
    return sg.RButton(
        "",
        image_filename=image,
        size=(1, 1),
        button_color=("white", color),
        pad=(0, 0),
        key=key,
    )


def redraw_board(window, board):
    for i in range(8):
        for j in range(8):
            color = "#B58863" if (i + j) % 2 else "#F0D9B5"
            piece_image = images[board[i][j]]
            elem = window.find_element(key=(i, j))
            elem.Update(
                button_color=("white", color),
                image_filename=piece_image,
            )


def convert_to_grid(move):
    move_str = move.uci()
    from_col = ord(move_str[0]) - ord("a")
    from_row = 8 - int(move_str[1])
    to_col = ord(move_str[2]) - ord("a")
    to_row = 8 - int(move_str[3])
    return (from_col, from_row, to_col, to_row)


def label_files():
    return [sg.T("     ")] + [
        sg.T("{}".format(a), pad=((23, 27), 0), font="Any 13") for a in "abcdefgh"
    ]


def label_rank(i):
    return sg.T(str(8 - i) + "   ", font="Any 13")


def layout_board():
    sg.ChangeLookAndFeel("GreenTan")
    psg_board = copy.deepcopy(initial_board)

    # the main board display layout
    board_layout = []

    # loop through the board row by row and create buttons with images
    for i in range(8):
        row = []
        for j in range(8):
            piece_image = images[psg_board[i][j]]
            row.append(render_square(piece_image, key=(i, j), location=(i, j)))

        # add the rank labels to the right side of the current row
        row.append(label_rank(i))
        board_layout.append(row)

    # add the file labels across bottom of board
    board_layout.append(label_files())

    board_controls = [
        [sg.Text("Move List", size=(16, 1))],
        [
            sg.Multiline(
                [], do_not_clear=True, autoscroll=True, size=(52, 8), key="_movelist_"
            )
        ],
    ]

    # the main window layout
    layout = [[sg.Column(board_layout), sg.Column(board_controls)]]

    return layout


def play_game(next_move):
    layout = layout_board()
    psg_board = copy.deepcopy(initial_board)

    window = sg.Window(
        "Chess",
        default_button_element_size=(12, 1),
        auto_size_buttons=False,
        icon="kingb.ico",
    ).Layout(layout)

    board = Board()
    moving_piece = False
    move_from = move_to = 0

    while not board.is_game_over():
        if board.turn == WHITE:
            moving_piece = False
            while True:
                button, value = window.Read()
                if button in (None, "Exit"):
                    exit()

                if type(button) is tuple:
                    # user has not clicked a piece to move yet,
                    # so wait for them to do so
                    if not moving_piece:
                        move_from = button
                        row, col = move_from

                        # get the piece on the square they clicked
                        piece = psg_board[row][col]
                        button_square = window.find_element(key=(row, col))
                        button_square.Update(button_color=("white", "red"))
                        moving_piece = True

                    # user has clicked a piece, so next click is
                    # the square to try to move the piece to
                    else:
                        move_to = button
                        row, col = move_to

                        if move_to == move_from:  # cancelled move
                            color = "#B58863" if (row + col) % 2 else "#F0D9B5"
                            button_square.Update(button_color=("white", color))
                            moving_piece = False
                            continue

                        # convert the grid index to the UCI format
                        picked_move = "{}{}{}{}".format(
                            "abcdefgh"[move_from[1]],
                            8 - move_from[0],
                            "abcdefgh"[move_to[1]],
                            8 - move_to[0],
                        )

                        picked_move = Move.from_uci(picked_move)

                        if board.is_legal(picked_move):
                            board.push(picked_move)
                        else:
                            print("Illegal move")
                            moving_piece = False
                            color = (
                                "#B58863"
                                if (move_from[0] + move_from[1]) % 2
                                else "#F0D9B5"
                            )
                            button_square.Update(button_color=("white", color))
                            continue

                        psg_board[move_from[0]][
                            move_from[1]
                        ] = BLANK  # place blank where piece was
                        psg_board[row][col] = piece  # place piece in the move-to square
                        redraw_board(window, psg_board)

                        window.find_element("_movelist_").Update(
                            picked_move.uci() + "\n", append=True
                        )

                        break
        else:
            fen = board.fen()
            best_move = next_move(fen)

            from_col, from_row, to_col, to_row = convert_to_grid(best_move)
            move_str = best_move.uci()

            window.find_element("_movelist_").Update(move_str + "\n", append=True)

            piece = psg_board[from_row][from_col]
            psg_board[from_row][from_col] = BLANK
            psg_board[to_row][to_col] = piece
            redraw_board(window, psg_board)

            board.push(best_move)
    sg.Popup("Game over!", "Thank you for playing")
