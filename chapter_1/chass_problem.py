import copy


class CheckersGame:
    def __init__(self):
        self.board = self._init_board()
        self.current_player = 1  # 1=己方(X/K), 2=对方(O/Q)

    def _init_board(self):
        board = [[0] * 8 for _ in range(8)]
        # 初始化己方棋子(1)在0-2行，对方棋子(2)在5-7行，仅放深色格
        for i in range(3):
            for j in range(8):
                if (i + j) % 2 == 1:
                    board[i][j] = 1
        for i in range(5, 8):
            for j in range(8):
                if (i + j) % 2 == 1:
                    board[i][j] = 2
        return board

    def print_board(self):
        print("  0 1 2 3 4 5 6 7")
        for i in range(8):
            print(i, end=" ")
            for j in range(8):
                p = self.board[i][j]
                print({0: ".", 1: "X", 2: "O", 3: "K", 4: "Q"}[p], end=" ")
            print()

    def get_legal_moves(self, player):
        moves, captures = [], []
        for i in range(8):
            for j in range(8):
                p = self.board[i][j]
                if (player == 1 and p in [1, 3]) or (player == 2 and p in [2, 4]):
                    caps = self._get_captures(i, j, player)
                    if caps:
                        captures.extend(caps)
                    else:
                        moves.extend(self._get_normals(i, j, player))
        return captures if captures else moves  # 有吃子必须吃

    def _get_captures(self, i, j, player):
        captures = []
        p, dirs = self.board[i][j], self._get_dirs(player, self.board[i][j])
        for di, dj in dirs:
            ni, nj = i + di, j + dj
            if 0 <= ni < 8 and 0 <= nj < 8:
                mid_p = self.board[ni][nj]
                if (player == 1 and mid_p in [2, 4]) or (player == 2 and mid_p in [1, 3]):
                    ti, tj = ni + di, nj + dj
                    if 0 <= ti < 8 and 0 <= tj < 8 and self.board[ti][tj] == 0:
                        captures.append(((i, j), (ti, tj)))
        return captures

    def _get_normals(self, i, j, player):
        moves = []
        dirs = self._get_dirs(player, self.board[i][j])
        for di, dj in dirs:
            ni, nj = i + di, j + dj
            if 0 <= ni < 8 and 0 <= nj < 8 and self.board[ni][nj] == 0:
                moves.append(((i, j), (ni, nj)))
        return moves

    def _get_dirs(self, player, piece):
        if piece in [1, 2]:  # 普通棋子
            return [(1, -1), (1, 1)] if player == 1 else [(-1, -1), (-1, 1)]
        else:  # 王棋
            return [(1, -1), (1, 1), (-1, -1), (-1, 1)]

    def make_move(self, move):
        (fi, fj), (ti, tj) = move
        p = self.board[fi][fj]
        self.board[ti][tj], self.board[fi][fj] = p, 0
        # 处理吃子
        if abs(ti - fi) == 2:
            self.board[(fi + ti) // 2][(fj + tj) // 2] = 0
        # 处理升变
        if p == 1 and ti == 7:
            self.board[ti][tj] = 3
        elif p == 2 and ti == 0:
            self.board[ti][tj] = 4
        self.current_player = 2 if self.current_player == 1 else 1

    def evaluate(self, player, weights):
        # 特征：[己方棋子数, 对方棋子数, 己方王数, 对方王数, 中心控制数]
        f = [0] * 5
        for i in range(8):
            for j in range(8):
                p = self.board[i][j]
                if p in [1, 3]:
                    f[0] += 1
                    f[2] += 1 if p == 3 else 0
                    f[4] += 1 if 2 <= i <= 5 and 2 <= j <= 5 else 0
                elif p in [2, 4]:
                    f[1] += 1
                    f[3] += 1 if p == 4 else 0
        score = sum(w * feat for w, feat in zip(weights, f))
        return score if player == 1 else -score

    def minimax(self, depth, max_depth, player, weights):
        if depth == max_depth or not self.get_legal_moves(player):
            return self.evaluate(player, weights), None
        legal_moves = self.get_legal_moves(player)
        best_move, best_score = None, -float('inf') if player == 1 else float('inf')
        for move in legal_moves:
            temp_board, temp_player = copy.deepcopy(self.board), self.current_player
            self.make_move(move)
            score, _ = self.minimax(depth + 1, max_depth, 2 if player == 1 else 1, weights)
            self.board, self.current_player = temp_board, temp_player
            if (player == 1 and score > best_score) or (player == 2 and score < best_score):
                best_score, best_move = score, move
        return best_score, best_move

    def self_play(self, weights, max_depth=2, max_moves=200):
        self.__init__()
        for _ in range(max_moves):
            player = self.current_player
            legal_moves = self.get_legal_moves(player)
            if not legal_moves:
                return 2 if player == 1 else 1  # 对方赢
            _, best_move = self.minimax(0, max_depth, player, weights)
            self.make_move(best_move if best_move else legal_moves[0])
        # 步数耗尽，比较棋子数
        count1 = sum(1 for row in self.board for p in row if p in [1, 3])
        count2 = sum(1 for row in self.board for p in row if p in [2, 4])
        return 1 if count1 > count2 else 2 if count2 > count1 else 0


# 演示：自对弈一局
if __name__ == "__main__":
    # 初始权重：[己方棋子+1, 对方棋子-1, 己方王+2, 对方王-2, 中心控制+0.5]
    initial_weights = [1.0, -1.0, 2.0, -2.0, 0.5]
    game = CheckersGame()

    print("=== 初始棋盘 ===")
    game.print_board()

    # 自对弈5步演示
    for step in range(100):
        player = game.current_player
        _, best_move = game.minimax(0,  6, player, initial_weights)
        if best_move:
            game.make_move(best_move)
            print(f"\n=== 第{step + 1}步：玩家{player}走棋 {best_move} ===")
            game.print_board()