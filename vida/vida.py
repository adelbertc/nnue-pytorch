import app as va

import argparse
from chess import Board
import chess
import chess.svg
import pynecone as pc
import tempfile

BOARD_SVG_SIZE = 500
STARTING_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
EVAL_DEPTH = 3

model = va.read_model("./data/nn-6877cd24400e.nnue")
model.eval()
model.cuda()

def render_board(fen):
    board = Board(fen)
    return chess.svg.board(board, size=BOARD_SVG_SIZE)

class State(pc.State):
    # A serializable way to track the state of the board
    fen = STARTING_FEN

    # The user's current input move (may be incomplete as they type)
    input_move = ""

    # The SVG code for rendering the board
    board_svg = render_board(fen)

    # Flag indicating when computer is thinking
    computer_thinking = False

    # Flag indicating if we should display illegal move dialog
    made_illegal_move = False

    # Flag indicating if there is checkmate on the board
    is_checkmate = False

    def commit_move(self, key=None):
        """
        Attempt the move entered.
        Called when user hits the submit button.
        """
        board = Board(self.fen)

        try:
            move = board.push_san(self.input_move)
            self.set_fen(board.fen())

            self.make_computer_move()
            self.input_move = ""
        except chess.IllegalMoveError:
            self.made_illegal_move = True
            self.input_move = ""


    def set_fen(self, fen):
        self.fen = fen
        self.render_board()

        self.check_checkmate()


    def check_checkmate(self):
        board = Board(self.fen)
        self.is_checkmate = board.is_checkmate()
        return self.is_checkmate


    def on_key_down(self, key):
        if key == "Enter":
            self.commit_move()
        elif key == "Backspace":
            self.input_move = self.input_move[:-1]


    def make_computer_move(self):
        if not self.check_checkmate():
            self.computer_thinking = True
            _, next_move = va.eval_position_with_search(model, self.fen, depth=EVAL_DEPTH)
            self.computer_thinking = False
            move = next_move[0]
            board = Board(self.fen)
            board.push(move)

            self.set_fen(board.fen())

    
    def set_move(self, s):
        self.input_move = s

    def render_board(self):
        self.board_svg = render_board(self.fen)

    def reset_board(self):
        self.set_fen(STARTING_FEN)
        self.check_checkmate()

    def toggle_illegal_move(self):
        self.made_illegal_move = not (self.made_illegal_move)


def index():
    board_state = pc.cond(
        State.computer_thinking,
        pc.circular_progress(is_indeterminate=True),
        pc.html(State.board_svg)
    )
    input_box = pc.input(
        value=State.input_move,
        on_change=State.set_move,
        on_key_down=State.on_key_down,
        placeholder="e.g. e4, Nf3, Rad1",
        width="20%"
    )
    submit_button = pc.button("Submit move", on_click=lambda: State.commit_move())
    reset_button = pc.button("Reset board", on_click=lambda: State.reset_board())
    buttons = pc.hstack(submit_button, reset_button)

    illegal_box_dialog = pc.alert_dialog(
        pc.alert_dialog_overlay(
            pc.alert_dialog_content(
                pc.alert_dialog_header("Error"),
                pc.alert_dialog_body("You made an illegal move, please enter in a legal move."),
                pc.alert_dialog_footer(pc.button("OK", on_click=State.toggle_illegal_move)),
            )
        ),
        is_open=State.made_illegal_move
    )

    checkmate_dialog = pc.alert_dialog(
        pc.alert_dialog_overlay(
            pc.alert_dialog_content(
                pc.alert_dialog_header("Checkmate"),
                pc.alert_dialog_body("Checkmate is on the board!"),
                pc.alert_dialog_footer(pc.button("OK", on_click=State.reset_board)),
            )
        ),
        is_open=State.is_checkmate
    )

    return pc.vstack(
        board_state,
        input_box,
        buttons,
        checkmate_dialog,
        illegal_box_dialog,
    )

app = pc.App(state=State)
app.add_page(index, title="Vida")
app.compile()
