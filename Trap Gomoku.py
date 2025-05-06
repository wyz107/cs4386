import random
from battle_base import read_ckbd, save_decision, MAX_M, MAX_N
from enum import IntEnum
import copy
import time
import sys

EMPTY = 0   
PLAYER_A = 1  
PLAYER_B = 2  
TRAP = 3      


DIRECTIONS = [
    (0, 1),   
    (1, 0),   
    (1, 1),   
    (1, -1)  
]

class PatternType(IntEnum):
    """棋型枚举类"""
    NONE = 0
    LIVE_ONE = 1       
    DEAD_ONE = 2       
    LIVE_TWO = 3      
    DEAD_TWO = 4       
    LIVE_THREE = 5     
    DEAD_THREE = 6     
    LIVE_FOUR = 7     
    DEAD_FOUR = 8      
    FIVE = 9           


PATTERN_SCORES = {
    'five': 100000,
    'open_four': 10000,
    'four': 1000,
    'open_three': 500,
    'three': 100,
    'open_two': 10,
    'two': 5,
    'double_three': 800,  
    'double_four': 5000,  
    'jump_four': 700,    
}
OPENING_BOOK = {

    "empty": (7, 7),  
 
    "center": (6, 6),  

    "corner_top_left": (8, 8),     # 对手在左上角(0,0)，我方下在靠近中心位置
    "corner_top_right": (8, 6),    # 对手在右上角(0,14)，我方下在靠近中心位置
    "corner_bottom_left": (6, 8),  # 对手在左下角(14,0)，我方下在靠近中心位置
    "corner_bottom_right": (6, 6), # 对手在右下角(14,14)，我方下在靠近中心位置
    
    # 特殊形态
    "swap_first": (8, 8),  # 互换先手策略，对手在中心周围，我方选择对称位置
    "block_three": None,   # 阻止对手形成活三，需要动态判断
    
    # 常见的开局路线
    "standard_line_1": {  # 标准开局路线1
        (7, 7): (6, 8),   # 对手中心点，我方左下
        (6, 8): (8, 6),   # 对手左下，我方右上
        (8, 6): (8, 8)    # 对手右上，我方右下
    },
    
    "standard_line_2": {  # 标准开局路线2
        (7, 7): (8, 8),   # 对手中心点，我方右下
        (8, 8): (6, 6),   # 对手右下，我方左上
        (6, 6): (8, 6)    # 对手左上，我方右上
    },
    
    # 应对常见棋形
    "respond_to_diagonal": (6, 8), # 应对对手在对角线上的两个子
    "respond_to_straight": (7, 9)  # 应对对手在直线上的两个子
}


class AdvancedTrapGomokuAI:
    """陷阱五子棋高级AI实现类"""
    
    def __init__(self):
        """初始化AI"""
        self.max_depth = 2  
        self.time_limit = 0.3 
        self.start_time = 0
        self.best_move = None
        self.transposition_table = {}
        self.move_ordering_cache = {}
        self.player = PLAYER_A  
        self.opponent = PLAYER_B 

    def is_time_up(self):
        """检查是否超时"""
        current_time = time.time()
        return current_time - self.start_time > self.time_limit

    def evaluate_board(self, board):
        """
        评估当前棋盘状态
        返回一个分数，分数越高对我方(PLAYER_A)越有利
        """
        my_score = self._evaluate_player(board, self.player)
        
        opponent_score = self._evaluate_player(board, self.opponent)
        
        return my_score - opponent_score * 1.1  
    
    def _evaluate_player(self, board, player):
        """评估指定玩家在当前棋盘的得分"""
        score = 0
        
        # 检查每个方向上的棋型
        for i in range(MAX_M):
            for j in range(MAX_N):
                if board[i][j] == player:
                    # 从该位置出发，检查四个方向的棋型
                    for dx, dy in DIRECTIONS:
                        pattern_score = self._evaluate_pattern(board, i, j, dx, dy, player)
                        score += pattern_score
        
        return score
    
    def _evaluate_pattern(self, board, row, col, dx, dy, player):
        """
        评估从(row,col)出发，沿(dx,dy)方向的棋型
        返回该棋型的得分
        """
        # 构建该方向的棋型字符串，用于模式匹配
        pattern = []
        for step in range(-4, 5):  # 考虑前后各4格，共9格
            r, c = row + step * dx, col + step * dy
            if 0 <= r < MAX_M and 0 <= c < MAX_N:
                if board[r][c] == player:
                    pattern.append('A')  
                elif board[r][c] == EMPTY:
                    pattern.append('E')  
                elif board[r][c] == TRAP:
                    pattern.append('T')  
                else:
                    pattern.append('B')  
            else:
                pattern.append('O')  
        
        pattern_str = ''.join(pattern)
        return self._score_pattern(pattern_str, player)
    
    # 在_score_pattern函数中添加对应的模式识别
    def _score_pattern(self, pattern_str, player):
    
        # 添加跳四模式
        if 'EAEAAE' in pattern_str or 'EAAEAE' in pattern_str:
            return PATTERN_SCORES['jump_four']
        # 五连
        if 'AAAAA' in pattern_str:
            return PATTERN_SCORES['five']
        
        # 活四
        if 'EAAAAE' in pattern_str:
            return PATTERN_SCORES['open_four']
        
        # 冲四 (单侧被封)
        if any(p in pattern_str for p in ['BAAAAE', 'EAAAAB', 'EAAAA', 'AAAAE','TAAAAE', 'EAAAAT']):
            return PATTERN_SCORES['four']
        
        # 活三
        if any(p in pattern_str for p in ['EAAAEE', 'EEAAAE', 'EAAEAE']):
            return PATTERN_SCORES['open_three']
        
        # 冲三 (单侧被封)
        if any(p in pattern_str for p in ['BAAAE', 'EAAAB', 'BAEAAE', 'EAEAAB','TAEAAE', 'EAEAAT','TAAAE', 'EAAAT']):
            return PATTERN_SCORES['three']
        
        # 活二
        if any(p in pattern_str for p in ['EEAAEE', 'EAEAE']):
            return PATTERN_SCORES['open_two']
        
        # 冲二
        if any(p in pattern_str for p in ['BAAE', 'EAAB','TAAE', 'EAAT']):
            return PATTERN_SCORES['two']
        
        return 0
    
    def get_valid_moves(self, board):

        """获取所有合法的落子位置（非陷阱、非已占用）"""
        moves = []
        occupied_positions = set()
        
        # 首先，找出所有已经有棋子的位置
        for i in range(MAX_M):
            for j in range(MAX_N):
                if board[i][j] in [PLAYER_A, PLAYER_B]:
                    occupied_positions.add((i, j))
        
        # 如果棋盘为空，就落在中心位置
        if not occupied_positions:
            center = MAX_M // 2
            return [(center, center)]
        
        # 考虑已有棋子周围的空位
        candidates = set()
        for i, j in occupied_positions:
            for di in range(-2, 3):  # 考虑周围2格内的位置
                for dj in range(-2, 3):
                    ni, nj = i + di, j + dj
                    if 0 <= ni < MAX_M and 0 <= nj < MAX_N and board[ni][nj] == EMPTY:
                        candidates.add((ni, nj))
        
        # 将候选位置转换为列表
        moves = list(candidates)
        
        # 如果没有合法的落子位置，就在所有空位中选择
        if not moves:
            for i in range(MAX_M):
                for j in range(MAX_N):
                    if board[i][j] == EMPTY:
                        moves.append((i, j))
        
        # 给每个可能的落子位置评分，用于排序
        move_scores = []
        for move in moves:
            i, j = move
            
            # 计算到中心的距离
            center_dist = abs(i - MAX_M // 2) + abs(j - MAX_N // 2)
            
            # 评估该位置的价值
            board[i][j] = self.player  # 尝试在该位置落子
            score = self.evaluate_board(board)
            board[i][j] = EMPTY     # 恢复
            
            move_scores.append((move, score, center_dist))
        
        # 按照评分和中心距离排序
        move_scores.sort(key=lambda x: (x[1], -x[2]), reverse=True)
        
        # 返回排序后的落子位置
        return [move for move, _, _ in move_scores[:15]]  # 只考虑前15个最优位置，提高效率
            # 进行初步评估和排序
        return self._order_moves(board, moves)
    
    def _order_moves(self, board, moves):
        """对移动进行启发式排序"""
        move_scores = []
        
        for move in moves:
            i, j = move
            score = 0
            
            # 1. 检查是否形成五连
            board[i][j] = self.player
            if self._check_win(board, i, j, self.player):
                board[i][j] = EMPTY
                return [move]  # 如果可以直接获胜，立即返回
            board[i][j] = EMPTY
            
            # 2. 检查是否阻止对手五连
            board[i][j] = self.opponent
            if self._check_win(board, i, j, self.opponent):
                score += 10000  # 高分优先考虑阻止对手获胜
            board[i][j] = EMPTY
            
            # 3. 基于棋型评估的得分
            board[i][j] = self.player
            my_score = self._evaluate_quick(board, i, j, self.player)
            board[i][j] = self.opponent
            opp_score = self._evaluate_quick(board, i, j, self.opponent)
            board[i][j] = EMPTY
            
            score += my_score + opp_score * 0.8
            
            # 4. 中心距离评分
            center_dist = abs(i - MAX_M // 2) + abs(j - MAX_N // 2)
            score -= center_dist * 0.5
            
            move_scores.append((move, score))
        
        # 按照分数排序
        move_scores.sort(key=lambda x: x[1], reverse=True)
        
        # 返回排序后的前N个落子位置
        return [move for move, _ in move_scores[:15]]

    def _check_win(self, board, row, col, player):
        """检查在(row, col)位置落子后是否形成五连"""
        for dx, dy in DIRECTIONS:
            count = 1  # 包括当前位置
            
            # 向一个方向查找
            for step in range(1, 5):
                r, c = row + step * dx, col + step * dy
                if 0 <= r < MAX_M and 0 <= c < MAX_N and board[r][c] == player:
                    count += 1
                else:
                    break
            
            # 向相反方向查找
            for step in range(1, 5):
                r, c = row - step * dx, col - step * dy
                if 0 <= r < MAX_M and 0 <= c < MAX_N and board[r][c] == player:
                    count += 1
                else:
                    break
            
            if count >= 5:
                return True
        
        return False


    def _evaluate_quick(self, board, row, col, player):
        """快速评估在(row, col)位置落子的价值"""
        score = 0
        for dx, dy in DIRECTIONS:
            pattern_score = self._evaluate_pattern(board, row, col, dx, dy, player)
            score += pattern_score
        return score

    # 在搜索函数开头添加超时检查
    def negamax_alpha_beta(self, board, depth, alpha, beta, color):
        # 检查是否超时
        if self.is_time_up():
            raise TimeoutError("搜索超时")
        
        # 生成棋盘的哈希值，用于置换表
        board_hash = self._hash_board(board)
        
        # 检查置换表
        if board_hash in self.transposition_table and self.transposition_table[board_hash]['depth'] >= depth:
            return self.transposition_table[board_hash]['score']
        
        # 终止条件：达到搜索深度或游戏结束
        if depth == 0 :
            score = color * self.evaluate_board(board)
            self.transposition_table[board_hash] = {'score': score, 'depth': depth}
            return score
        
        max_score = float('-inf')
        best_move = None
        
        # 获取并排序可能的落子位置
        valid_moves = self.get_valid_moves(board)
        
        current_player = self.player if color == 1 else self.opponent
        
        for move in valid_moves:
            i, j = move
            
            # 尝试落子
            board[i][j] = current_player
            
            # 递归搜索，注意color取反
            score = -self.negamax_alpha_beta(board, depth - 1, -beta, -alpha, -color)
            
            # 恢复棋盘
            board[i][j] = EMPTY
            
            # 更新最大分数
            if score > max_score:
                max_score = score
                best_move = move
                
                # 如果在最顶层，记录最佳落子位置
                if depth == self.max_depth:
                    self.best_move = move
            
            # Alpha-Beta剪枝
            alpha = max(alpha, max_score)
            if alpha >= beta:
                break
        
        # 存储到置换表
        self.transposition_table[board_hash] = {'score': max_score, 'depth': depth}
        
        return max_score
    
    def _hash_board(self, board):
        """生成棋盘的简单哈希值"""
        hash_val = 0
        for i in range(MAX_M):
            for j in range(MAX_N):
                hash_val = hash_val * 4 + board[i][j]
        return hash_val
    

    def _is_opening(self, board):
        """判断是否处于开局阶段"""
        stone_count = 0
        for i in range(MAX_M):
            for j in range(MAX_N):
                if board[i][j] in [PLAYER_A, PLAYER_B]:
                    stone_count += 1
        return stone_count <= 4  # 前4步棋认为是开局

    def _get_opening_move(self, board):
        """从开局库中获取走法"""
        # 空棋盘
        empty = True
        for i in range(MAX_M):
            for j in range(MAX_N):
                if board[i][j] != EMPTY and board[i][j] != TRAP:
                    empty = False
                    break
            if not empty:
                break
        
        if empty:
            return OPENING_BOOK["empty"]
        
        # 对手在中心点
        center_x, center_y = MAX_M // 2, MAX_N // 2
        if board[center_x][center_y] == self.opponent:
            return OPENING_BOOK["center"]
        
        # 对手在四个角落的应对
        if board[0][0] == self.opponent:
            return OPENING_BOOK["corner_top_left"]
        if board[0][MAX_N-1] == self.opponent:
            return OPENING_BOOK["corner_top_right"]
        if board[MAX_M-1][0] == self.opponent:
            return OPENING_BOOK["corner_bottom_left"]
        if board[MAX_M-1][MAX_N-1] == self.opponent:
            return OPENING_BOOK["corner_bottom_right"]
        
        # 特殊形态
        if board[MAX_M//2-1][MAX_N//2] == self.opponent and board[MAX_M//2+1][MAX_N//2] == self.opponent:
            return OPENING_BOOK["swap_first"]
        
        # 常见的开局路线
        if (MAX_M//2, MAX_N//2) in OPENING_BOOK["standard_line_1"]:
            if board[MAX_M//2][MAX_N//2] == self.opponent:
                return OPENING_BOOK["standard_line_1"][(MAX_M//2, MAX_N//2)]
        if (MAX_M//2, MAX_N//2) in OPENING_BOOK["standard_line_2"]:
            if board[MAX_M//2][MAX_N//2] == self.opponent:
                return OPENING_BOOK["standard_line_2"][(MAX_M//2, MAX_N//2)]
        
        # 应对常见棋形
        for i in range(MAX_M):
            for j in range(MAX_N):
                if board[i][j] == self.opponent and (i+1 < MAX_M and board[i+1][j+1] == self.opponent):
                    return OPENING_BOOK["respond_to_diagonal"]
                if board[i][j] == self.opponent and (i+1 < MAX_M and board[i+1][j] == self.opponent):
                    return OPENING_BOOK["respond_to_straight"]
        
        # 其他开局模式...
        return None

    def find_best_move(self, board):
        """找到最佳落子位置"""
        # 清空缓存
        self.transposition_table = {}
        self.best_move = None
        self.start_time = time.time()
        
        # 快速检查：是否有立即获胜或防守的位置
        quick_win = self._find_quick_win(board)
        if quick_win:
            return quick_win
            
        # 检查是否可以使用开局库
        if self._is_opening(board):
            opening_move = self._get_opening_move(board)
            if opening_move and self._is_valid_move(board, opening_move):
                return opening_move

        # 优化：只进行启发式评估，不做深度搜索
        # 如果棋盘已经有很多子，我们降低搜索深度以避免超时
        piece_count = 0
        for i in range(MAX_M):
            for j in range(MAX_N):
                if board[i][j] != EMPTY:
                    piece_count += 1
        
        # 根据棋盘上棋子数量动态调整搜索深度
        if piece_count > 20:
            max_depth = 2  # 棋子多时降低搜索深度
        else:
            max_depth = min(2, self.max_depth)  # 使用较低的搜索深度
            
        # 迭代加深搜索，使用更严格的时间控制
        try:
            for depth in range(1, max_depth + 1):
                # 如果已经用了超过40%的时间，则不再增加深度
                if time.time() - self.start_time > self.time_limit * 0.4:
                    break
                # 设置足够大的Alpha-Beta窗口
                self.negamax_alpha_beta(board, depth, float('-inf'), float('inf'), 1)
                # 每次迭代后检查是否即将超时
                if time.time() - self.start_time > self.time_limit * 0.8:
                    break
        except TimeoutError:
            # 如果超时，就使用上一次的结果
            pass
        
        # 如果深度搜索超时或没有找到好的落子点，使用快速启发式方法
        if self.best_move is None or not self._is_valid_move(board, self.best_move):
            valid_moves = self.get_valid_moves(board)
            # 使用简单评估函数
            best_score = float('-inf')
            for move in valid_moves[:min(10, len(valid_moves))]:  # 只考虑前10个可能的移动
                if self._is_valid_move(board, move):
                    x, y = move
                    board[x][y] = self.player
                    score = self._simple_evaluate(board, x, y)
                    board[x][y] = EMPTY
                    
                    if score > best_score:
                        best_score = score
                        self.best_move = move
                    
                    # 检查是否即将超时
                    if time.time() - self.start_time > self.time_limit * 0.9:
                        break
        
        # 最后的安全检查
        if self.best_move and not self._is_valid_move(board, self.best_move):
            # 如果仍然没有找到有效的移动，寻找任何空位
            self.best_move = self._find_any_empty_position(board)
            
        return self.best_move
        
    def _simple_evaluate(self, board, x, y):
        """简单的局面评估函数，用于快速评估落子位置的好坏"""
        score = 0
        # 检查我方是否可能获胜
        if self._check_win(board, x, y, self.player):
            return 10000
            
        # 检查八个方向
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
        
        for dx, dy in directions:
            # 统计两个方向的连子情况
            count_own = 1  # 包括当前位置
            count_empty = 0
            
            # 正方向
            nx, ny = x + dx, y + dy
            while 0 <= nx < MAX_M and 0 <= ny < MAX_N:
                if board[nx][ny] == self.player:
                    count_own += 1
                elif board[nx][ny] == EMPTY:
                    count_empty += 1
                    break
                else:
                    break
                nx, ny = nx + dx, ny + dy
            
            # 反方向
            nx, ny = x - dx, y - dy
            while 0 <= nx < MAX_M and 0 <= ny < MAX_N:
                if board[nx][ny] == self.player:
                    count_own += 1
                elif board[nx][ny] == EMPTY:
                    count_empty += 1
                    break
                else:
                    break
                nx, ny = nx - dx, ny - dy
            
            # 评分规则
            if count_own >= 5:
                score += 10000  # 五连胜
            elif count_own == 4 and count_empty >= 1:
                score += 1000   # 活四
            elif count_own == 3 and count_empty >= 2:
                score += 100    # 活三
            elif count_own == 2 and count_empty >= 2:
                score += 10     # 活二
        
        return score

    def _is_valid_move(self, board, move):
        """检查移动是否有效（在边界内且位置为空）"""
        if move is None:
            return False
            
        i, j = move
        if 0 <= i < MAX_M and 0 <= j < MAX_N:
            return board[i][j] == EMPTY
        return False
        
    def _find_any_empty_position(self, board):
        """找到任何一个空位"""
        for i in range(MAX_M):
            for j in range(MAX_N):
                if board[i][j] == EMPTY:
                    return (i, j)
        return None

    def _find_quick_win(self, board):
        """快速查找制胜或防守的位置，不需要深度搜索"""
        # 首先，检查自己是否有可能获胜的四连
        for i in range(MAX_M):
            for j in range(MAX_N):
                if board[i][j] == EMPTY:
                    board[i][j] = self.player
                    if self._check_win(board, i, j, self.player):
                        board[i][j] = EMPTY
                        return (i, j)
                    board[i][j] = EMPTY
        
        # 然后，检查对手是否有可能获胜的四连
        for i in range(MAX_M):
            for j in range(MAX_N):
                if board[i][j] == EMPTY:
                    board[i][j] = self.opponent
                    if self._check_win(board, i, j, self.opponent):
                        board[i][j] = EMPTY
                        return (i, j)
                    board[i][j] = EMPTY
        
        return None

def play_games(step, rival_decision_x, rival_decision_y):

    # 读取当前棋盘状态
    board = read_ckbd(step-1)
    
    try:
        # 确定当前玩家，基于步数
        current_player = PLAYER_A if step % 2 == 1 else PLAYER_B
     
        
        if step > 1 and rival_decision_x >= 0 and rival_decision_y >= 0:
            # 确保对手的落子坐标在合理范围内
            if 0 <= rival_decision_x < MAX_M and 0 <= rival_decision_y < MAX_N:
                # 记录对手的落子位置，但不要修改棋盘，因为棋盘已经包含了对手的落子
                opponent_pos = (rival_decision_x, rival_decision_y)
                
        ai = AdvancedTrapGomokuAI()
        
        if step % 2 == 1:  
            ai.player = PLAYER_A
            ai.opponent = PLAYER_B
        else: 
            ai.player = PLAYER_B
            ai.opponent = PLAYER_A
        
        best_move = ai.find_best_move(board)
        
        if best_move:
            x, y = best_move
           
            
            # 确保位置确实为空
            if board[x][y] == EMPTY:
                save_decision(x, y)
               
                return
            else:
               
                for i in range(MAX_M):
                    for j in range(MAX_N):
                        if board[i][j] == EMPTY:
                        
                            save_decision(i, j)
                            return
        
       
        save_decision(MAX_M//2, MAX_N//2)
    except Exception as e:
        print(f"处理AI决策时发生错误: {str(e)}")
        # 出错时尝试使用中心位置
        save_decision(MAX_M//2, MAX_N//2)