from modules.cboard import CBoard
from modules.mcts import MCTS
from modules.utils import count,display,num2pos,pos2num,BLACK,WHITE,BOARD
import onnxruntime
import argparse
import subprocess

###コマンドライン引数#########################################################################
parser = argparse.ArgumentParser(description='Edax')
parser.add_argument('--model', type=str, required=True, help='学習済みモデルパス（onnx）')
parser.add_argument('--playout', type=int, default=100, help='プレイアウト数')
parser.add_argument('--batch_size', type=int, default=4, help='バッチサイズ')
parser.add_argument('--second', action="store_true", help='後攻になる')
parser.add_argument('--human_play', action="store_true", help='自分で戦う')
############################################################################################

def board2text(board):
    text = ""
    for point in board.board[BOARD]:
        if point == BLACK:
            text += "*"
        elif point == WHITE:
            text += "O"
        else:
            text += "-"
    if board.turn == BLACK:
        text += "*"
    else:
        text += "O"
    
    with open("board.txt" , "w") as f:
        f.write(text)
    return 


def main(args):
    board = CBoard()
    display(board)
    session = onnxruntime.InferenceSession(args.model,providers=['CPUExecutionProvider'])
    mcts = MCTS(board,session,args.playout,batch_size=args.batch_size,perfect=5)
    
    player =  WHITE if args.second else BLACK
    
    while True:
        move = 0 if not args.human_play else None
        if mcts.original_board.turn == player:
            check = True
            while move not in mcts.original_board.legal_moves and move != 0:
                if check:
                    message = "指し手を入力してください（0とするとコンピュータに任せます）"
                else:
                    message = "合法手ではないようです。もう一度入力してください（0とするとコンピュータに任せます）"
                check = False
                move = input(message)
                try:
                    move = pos2num(move) if move != "0" else 0
                except:
                    move = "no"
            if move == 0:
                print("コンピュータが指しますよ")
                move,eval = mcts.move(re_eval=True)
                print(f"指し手：{num2pos(move)},評価値：{eval}")
            else:
                mcts.move_enemy(move)
            display(mcts.original_board)
            if len(mcts.original_board.legal_moves) == 0:
                print("対局終了でーす。")
                black,white = count(mcts.original_board) 
                print(f"黒:{black},白:{white}")
                return
        else:
            print("EDAXの手番ですよ")
            board2text(mcts.original_board)
            #https://qiita.com/y-tetsu/items/2d5a199e401aa846891f から引用
            cmd = ["./edax-4.4" ,"-solve" ,"./board.txt"]
            output_str = subprocess.run(cmd, capture_output=True, text=True).stdout.split('\n')[2][57:].split()[0].lower()
            mcts.move_enemy(pos2num(output_str))
            display(mcts.original_board)
            if len(mcts.original_board.legal_moves) == 0:
                print("対局終了でーす。")
                black,white = count(mcts.original_board) 
                print(f"黒:{black},白:{white}")
                return 

if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
