import random
import itertools
import math
from battle_base import read_ckbd, save_decision, MAX_M, MAX_N



def get_moves_for_one_pice(board, pice, my_positions, enemy_positions):
    x, y = pice
    moves = []
    src_shape, src_role = board[x][y]
    if src_shape not in ('O', 'S') or src_role not in (1, 2):
        return moves

    directions = [(1,0), (-1,0), (0,1), (0,-1), (1,1), (1,-1), (-1,1), (-1,-1)]
    move_dirs = directions if src_shape == 'O' else directions[:4]

    for dx, dy in move_dirs:
        nx, ny = x + dx, y + dy
        if 0 <= nx < MAX_M and 0 <= ny < MAX_N and board[nx][ny][1] == 0:
            moves.append(((x, y), (nx, ny)))

    for dx, dy in directions:
        bx, by = x - dx, y - dy
        while 0 <= bx < MAX_M and 0 <= by < MAX_N:
            if (bx, by) not in my_positions:
                if board[bx][by][1] != 0: break
                bx -= dx
                by -= dy
                continue
            cx, cy = x + dx, y + dy
            while 0 <= cx < MAX_M and 0 <= cy < MAX_N:
                if (cx, cy) in my_positions: break
                if (cx, cy) in enemy_positions:
                    clear = True
                    checkx, checky = bx + dx, by + dy
                    while (checkx, checky) != (cx, cy):
                        if (checkx, checky) == (x, y):
                            checkx += dx
                            checky += dy
                            continue
                        if not (0 <= checkx < MAX_M and 0 <= checky < MAX_N) or board[checkx][checky][1] != 0:
                            clear = False
                            break
                        checkx += dx
                        checky += dy
                    if clear:
                        moves.append(((x, y), (cx, cy)))
                    break
                elif board[cx][cy][1] != 0:
                    break
                cx += dx
                cy += dy
            break
    return moves



def evaluate_move(board, move, enemy_positions):
    (x1, y1), (x2, y2) = move
    role = board[x1][y1][1]
    score = 0
    
    old_positions = [(i, j) for i in range(MAX_M) for j in range(MAX_N) if board[i][j][1] == role]
    my_positions = [(i, j) for i in range(MAX_M) for j in range(MAX_N) if board[i][j][1] == role and (i, j) != (x1, y1)]
    my_positions.append((x2, y2))
    enemy_positions_after = [pos for pos in enemy_positions if pos != (x2, y2)]

    new_board = [[cell.copy() for cell in row] for row in board]
    new_board[x2][y2][1] = role
    new_board[x1][y1][1] = 0

    if (x2, y2) in enemy_positions and len(enemy_positions) <= 4:
        return 10000

    if (x2, y2) in enemy_positions:
        if not is_under_threat(new_board, (x2, y2), role ^ 3):
            score += 3000
        else:
            score += 2000


    # 如果被攻击的棋子可以攻击我方这颗棋子则不计算分数
    if not is_under_threat(new_board, (x2, y2), role ^ 3):
        threats,attacked_important = count_threats(new_board, (x2, y2), my_positions, enemy_positions_after, role)
        score += threats * 800 
        

    if is_under_threat(new_board, (x2, y2), role ^ 3):
        danger_penalty = 750 * (7-len(my_positions))
        score -= danger_penalty

    if is_under_threat(board, (x1, y1), role ^ 3) and not is_under_threat(new_board, (x2, y2), role ^ 3):
        if len(my_positions) <= 5:
            score += 6000
    
    # --- 第二梯队的分数 ---
    # 如果这一步移动是向己方靠拢加分
    if total_euclidean_distance(my_positions) < total_euclidean_distance(old_positions):
        score += 100

    # 如果这一步可以保护我方棋子则加分

    #如果这步棋盘中间靠拢加分
    if (abs(x2-4)+abs(y2-4)) < (abs(x1-4)+abs(y1-4)):
        score += 80

    return score


def total_euclidean_distance(positions):
    """
    计算传入的所有棋子位置两两之间的欧几里得距离总和。

    参数：
        positions: List of (x, y) 坐标的列表

    返回：
        所有 (i, j) 之间的欧几里得距离之和
    """
    total = 0.0
    for (x1, y1), (x2, y2) in itertools.combinations(positions, 2):
        dist = math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
        total += dist
    return total



def can_capture_by_firing(B, A, C, board, my_role, enemy_role):
    """
    判断是否可以由 B 发射出去攻击 C，A 是协助棋子。
    B 是移动的发射棋子，A 是己方协助者，C 是敌方目标。
    条件：ABC 共线，B 在 A 和 C 之间，路径无阻挡。
    """
    ax, ay = A
    bx, by = B
    cx, cy = C

    # 必须共线（向量平行）
    v1x, v1y = ax - bx, ay - by
    v2x, v2y = cx - bx, cy - by
    if v1x * v2y != v1y * v2x:
        return False

    # B 必须在 A 和 C 之间：判断 B 是否在 AC 线段上
    def between(a, b, c):
        return min(a, c) < b < max(a, c)

    if not (between(ax, bx, cx) or between(ay, by, cy)):
        return False

    # 检查身份合法
    if board[ax][ay][1] != my_role:
        return False
    if board[bx][by][1] != my_role:
        return False
    if board[cx][cy][1] != enemy_role:
        return False

    # 单步方向向量（肯定可以整除，不用担心，因为有三个棋子共线约束）
    dx = cx - ax
    dy = cy - ay
    steps = max(abs(dx), abs(dy))
    dx = dx // steps
    dy = dy // steps

    # 检查路径 A ➝ B ➝ C 中间是否无阻挡（不包括 A, B, C）
    px, py = ax + dx, ay + dy
    while (px, py) != C:
        if (px, py) != B and board[px][py][1] != 0:
            return False
        px += dx
        py += dy

    return True



def count_threats(board, pos, my_positions, enemy_positions, my_role):
    """
    计算当前棋子 pos 作为攻击发射者，可以通过协助者 A 攻击到的敌方棋子数 C。
    """
    threats = 0
    attacked_important = 0
    for A in my_positions:
        if A == pos:
            continue
        for C in enemy_positions:
            if can_capture_by_firing(pos, A, C, board, my_role, board[C[0]][C[1]][1]):
                threats += 1
            for E in enemy_positions:
                if E == C:
                    continue
                for D in enemy_positions:
                    if D == C or D == E:
                        continue

                    # 检查 E, C, D 是否共线
                    v1x, v1y = E[0] - C[0], E[1] - C[1]
                    v2x, v2y = D[0] - C[0], D[1] - C[1]
                    if v1x * v2y != v1y * v2x:
                        continue

                    # 检查 C 是否在 E 和 D 之间（但不是中间）
                    def between(a, b, c):
                        return min(a, c) < b < max(a, c)

                    if between(E[0], C[0], D[0]) or between(E[1], C[1], D[1]):
                        continue

                    # 如果满足条件，执行相关逻辑
                    attacked_important += 1
    return threats,attacked_important



def is_under_threat(board, pos, enemy_role):
    """
    判断当前位置 pos 是否会被敌方玩家捕获。

    参数：
        board       : 当前棋盘
        pos         : 我方棋子的位置 (x, y)
        enemy_role  : 敌方编号（1或2）

    返回：
        True  - 如果存在敌方棋子能攻击该位置
        False - 否则
    """
    x, y = pos
    my_role = board[x][y][1]

    # 收集敌方棋子位置
    enemy_positions = [
        (i, j) for i in range(MAX_M) for j in range(MAX_N)
        if board[i][j][1] == enemy_role
    ]

    for B in enemy_positions:       # B 为准备发射的敌方棋子
        for A in enemy_positions:   # A 为协助者
            if A == B:
                continue
            if can_capture_by_firing(B, A, pos, board, enemy_role, my_role):
                return True
    return False



def choose_best_move(board, my_positions, enemy_positions):
    best_move = None
    best_score = float('-inf')
    all_moves = []
    
    for pice in my_positions:
        moves = get_moves_for_one_pice(board, pice, my_positions, enemy_positions)
        all_moves += moves
        for move in moves:
            score = evaluate_move(board, move, enemy_positions)
            if score > best_score:
                best_score = score
                best_move = move
    return best_move,best_score



def play_games(step: int) -> None:
    playerA_positions, playerB_positions, board = read_ckbd(step - 1)
    player_id = 1 if step % 2 == 1 else 2
    my_positions = playerA_positions if player_id == 1 else playerB_positions
    enemy_positions = playerB_positions if player_id == 1 else playerA_positions
    best_move,best_score = choose_best_move(board, my_positions, enemy_positions)
    pice_from, pice_to = best_move
    print("show my score:",best_score)
    print("show my move:", pice_from, pice_to)
    save_decision(pice_from[0], pice_from[1], pice_to[0], pice_to[1])



