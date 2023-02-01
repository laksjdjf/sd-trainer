from modules.cboard import CBoard
from modules.mcts import MCTS
from modules.utils import BOARD,BLACK,WHITE,num2pos,pos2num
import time
import onnxruntime
import argparse
from tqdm import tqdm
import random
import re
import subprocess

###コマンドライン引数#########################################################################
parser = argparse.ArgumentParser(description='StableDiffusionの訓練コード')
parser.add_argument('model1', type=str, help='学習済みモデルパス1（onnx）')
parser.add_argument('--playout', type=int, default=100, help='プレイアウト数')
parser.add_argument('--batch_size', type=int, default=4, help='バッチサイズ')
parser.add_argument('--level', type=str, default=5, help='Edaxの探索深さ')
parser.add_argument('--num_plays', type=int, default=100, help='試合数')
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
    epsilon = 0.9
    model_time = 0
    edax_time = 0
    model_win = 0
    edax_win = 0

    black_win = 0
    white_win = 0
    
    session = onnxruntime.InferenceSession(args.model1,providers=['CUDAExecutionProvider','CPUExecutionProvider'])
    
    #プログレスバー
    progress_bar = tqdm(range(args.num_plays), desc="Total Steps", leave=False)
    for i in range(args.num_plays):
        board = CBoard()
        mcts = MCTS(board,session,args.playout,batch_size=args.batch_size,perfect=5)
        
        while True:
            if board.turn == [1,-1][i%2]:
                start_time = time.perf_counter()
                if random.random() <= epsilon:
                    move = mcts.move(0)
                else:
                    move = mcts.move(1)
                end_time = time.perf_counter()
                model_time += end_time - start_time    
            else:
                board2text(mcts.original_board)
                #https://qiita.com/y-tetsu/items/2d5a199e401aa846891f から引用
                cmd = ["./edax-4.4","-l",str(args.level),"-solve" ,"./board.txt"]
                out = subprocess.run(cmd, capture_output=True, text=True).stdout
                move = out.split('\n')[2][57:].split()[0].lower()
                
                step_time = out.split('\n')[4]
                step_time = re.findall(r"\d+\.\d+",step_time)[0]
                edax_time += float(step_time)
                move = pos2num(move)
                mcts.move_enemy(move)
                
            result = board.move(move)

            if result == 1:
                num = board.board[BOARD].sum()
                if [1,-1][i%2] * num > 0:
                    model_win += 1
                elif num != 0:
                    edax_win += 1
                if num > 0:
                    black_win += 1
                elif num < 0:
                      white_win += 1
                break
        #プログレスバー更新
        logs={"model_win":model_win,"edax_win":edax_win,"model_time":model_time,"edax_time":edax_time}
        progress_bar.update(1)
        progress_bar.set_postfix(logs)
        
    print("")
    print(f"{model_win}勝{edax_win}敗！！！")
    print(f"先攻：{black_win}勝,後攻：{white_win}勝")
    print(f"モデル思考時間：{model_time}")
    print(f"edax思考時間：{edax_time}")

if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
