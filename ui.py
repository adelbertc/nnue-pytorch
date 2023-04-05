"""
Copied and adapted from the chess demo code for PySimpleGUI.

https://github.com/PySimpleGUI/PySimpleGUI
"""

import argparse
import chess
import chess.pgn
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

initial_board = [[ROOKB, KNIGHTB, BISHOPB, QUEENB, KINGB, BISHOPB, KNIGHTB, ROOKB],
                 [PAWNB, ] * 8,
                 [BLANK, ] * 8,
                 [BLANK, ] * 8,
                 [BLANK, ] * 8,
                 [BLANK, ] * 8,
                 [PAWNW, ] * 8,
                 [ROOKW, KNIGHTW, BISHOPW, QUEENW, KINGW, BISHOPW, KNIGHTW, ROOKW]]

blank = os.path.join(CHESS_PATH, 'blank.png')
bishopB = os.path.join(CHESS_PATH, 'nbishopb.png')
bishopW = os.path.join(CHESS_PATH, 'nbishopw.png')
pawnB = os.path.join(CHESS_PATH, 'npawnb.png')
pawnW = os.path.join(CHESS_PATH, 'npawnw.png')
knightB = os.path.join(CHESS_PATH, 'nknightb.png')
knightW = os.path.join(CHESS_PATH, 'nknightw.png')
rookB = os.path.join(CHESS_PATH, 'nrookb.png')
rookW = os.path.join(CHESS_PATH, 'nrookw.png')
queenB = os.path.join(CHESS_PATH, 'nqueenb.png')
queenW = os.path.join(CHESS_PATH, 'nqueenw.png')
kingB = os.path.join(CHESS_PATH, 'nkingb.png')
kingW = os.path.join(CHESS_PATH, 'nkingw.png')

images = {BISHOPB: bishopB, BISHOPW: bishopW, PAWNB: pawnB, PAWNW: pawnW, KNIGHTB: knightB, KNIGHTW: knightW,
          ROOKB: rookB, ROOKW: rookW, KINGB: kingB, KINGW: kingW, QUEENB: queenB, QUEENW: queenW, BLANK: blank}

def render_square(image, key, location):
    if (location[0] + location[1]) % 2:
        color = '#B58863'
    else:
        color = '#F0D9B5'
    return sg.RButton('', image_filename=image, size=(1, 1), button_color=('white', color), pad=(0, 0), key=key)

def redraw_board(window, board):
    for i in range(8):
        for j in range(8):
            color = '#B58863' if (i + j) % 2 else '#F0D9B5'
            piece_image = images[board[i][j]]
            elem = window.find_element(key=(i, j))
            elem.Update(button_color=('white', color),
                        image_filename=piece_image, )

def play_game(next_move):
    # sg.SetOptions(margins=(0,0))
    sg.ChangeLookAndFeel('GreenTan')
    # create initial board setup
    psg_board = copy.deepcopy(initial_board)
    # the main board display layout
    board_layout = [[sg.T('     ')] + [sg.T('{}'.format(a), pad=((23, 27), 0), font='Any 13') for a in 'abcdefgh']]
    # loop though board and create buttons with images
    for i in range(8):
        row = [sg.T(str(8 - i) + '   ', font='Any 13')]
        for j in range(8):
            piece_image = images[psg_board[i][j]]
            row.append(render_square(piece_image, key=(i, j), location=(i, j)))
        row.append(sg.T(str(8 - i) + '   ', font='Any 13'))
        board_layout.append(row)
    # add the labels across bottom of board
    board_layout.append([sg.T('     ')] + [sg.T('{}'.format(a), pad=((23, 27), 0), font='Any 13') for a in 'abcdefgh'])

    board_controls = [[sg.Text('Move List')], [sg.Multiline([], do_not_clear=True, autoscroll=True, size=(15, 10), key='_movelist_')]]

    board_tab = [[sg.Column(board_layout)]]

    # the main window layout
    layout = [[sg.Column(board_tab)],
              [sg.Column(board_controls)]]

    window = sg.Window('Chess',
                       default_button_element_size=(12, 1),
                       auto_size_buttons=False,
                       icon='kingb.ico').Layout(layout)

    board = chess.Board()
    move_count = 1
    move_state = move_from = move_to = 0
    # ---===--- Loop taking in user input --- #
    while not board.is_game_over():

        if board.turn == chess.WHITE:
            # human_player(board)
            move_state = 0
            while True:
                button, value = window.Read()
                if button in (None, 'Exit'):
                    exit()

                if type(button) is tuple:
                    if move_state == 0:
                        move_from = button
                        row, col = move_from
                        piece = psg_board[row][col]  # get the move-from piece
                        button_square = window.find_element(key=(row, col))
                        button_square.Update(button_color=('white', 'red'))
                        move_state = 1
                    elif move_state == 1:
                        move_to = button
                        row, col = move_to
                        if move_to == move_from:  # cancelled move
                            color = '#B58863' if (row + col) % 2 else '#F0D9B5'
                            button_square.Update(button_color=('white', color))
                            move_state = 0
                            continue

                        picked_move = '{}{}{}{}'.format('abcdefgh'[move_from[1]], 8 - move_from[0],
                                                        'abcdefgh'[move_to[1]], 8 - move_to[0])

                        if picked_move in [str(move) for move in board.legal_moves]:
                            board.push(chess.Move.from_uci(picked_move))
                        else:
                            print('Illegal move')
                            move_state = 0
                            color = '#B58863' if (move_from[0] + move_from[1]) % 2 else '#F0D9B5'
                            button_square.Update(button_color=('white', color))
                            continue

                        psg_board[move_from[0]][move_from[1]] = BLANK  # place blank where piece was
                        psg_board[row][col] = piece  # place piece in the move-to square
                        redraw_board(window, psg_board)
                        move_count += 1

                        window.find_element('_movelist_').Update(picked_move + '\n', append=True)

                        break
        else:
            fen = board.fen()
            best_move = next_move(fen)

            move_str = best_move.uci()
            from_col = ord(move_str[0]) - ord('a')
            from_row = 8 - int(move_str[1])
            to_col = ord(move_str[2]) - ord('a')
            to_row = 8 - int(move_str[3])

            window.find_element('_movelist_').Update(move_str + '\n', append=True)

            piece = psg_board[from_row][from_col]
            psg_board[from_row][from_col] = BLANK
            psg_board[to_row][to_col] = piece
            redraw_board(window, psg_board)

            board.push(best_move)
            move_count += 1
    sg.Popup('Game over!', 'Thank you for playing')
