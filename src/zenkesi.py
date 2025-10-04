import copy
import random
import sys
import pygame
from collections import deque
from collections import namedtuple

# 定数
CELL_SIZE = 40
COLUMNS = 6
ROWS = 14
NEXT_AREA_WIDTH = 200
SCREEN_WIDTH = CELL_SIZE * COLUMNS + NEXT_AREA_WIDTH + CELL_SIZE * COLUMNS
SCREEN_HEIGHT = CELL_SIZE * ROWS
BEST_FIELD_OFFSET_X = CELL_SIZE * COLUMNS + NEXT_AREA_WIDTH

# ぷよの色（初期配置は単一色、追加配置ではタプルの1要素として扱う）
PUYO_COLORS = [
    (255, 0, 0),
    (0, 255, 0),
    (0, 0, 255),
    (255, 255, 0)
]

# 盤面初期化（ここでは全セルが None）
field = [[None for _ in range(COLUMNS)] for _ in range(ROWS)]
undo_stack = deque()

# --- ヘルパー関数：セルから色成分を取り出す ---
def get_cell_color(cell):
    """
    cellが (color, iterationID) のタプルの場合は色部分を返す。
    もともと単一色の表現の場合もそのまま返す。
    """
    return cell[0] if isinstance(cell, tuple) else cell

# --- デバッグ出力用関数 ---
def color_to_letter(cell):
    if cell is None:
        return "."
    c = get_cell_color(cell)
    mapping = {
        (255, 0, 0): "R",
        (0, 255, 0): "G",
        (0, 0, 255): "B",
        (255, 255, 0): "Y"
    }
    return mapping.get(c, "?")

def print_board(board, label):
    print(f"--- {label} ---")
    for y, row in enumerate(board):
        row_str = ""
        for x, cell in enumerate(row):
            row_str += color_to_letter(cell) + " "
        print(row_str)
    print()

# --- 追加配置用 ロガークラスの定義 ---
class PlacementLogger:
    def __init__(self):
        self.entries = []

    # --- ベース候補ごとの見出し用ログ ---
    def header(self, iteration, x, y, color):
        c = color_to_letter((color, iteration))
        self.entries.append(f"\n[Iter{iteration}] ==== ベース候補: 列 {x}, 行 {y}, 色 {c} ==== ")

    def attempt(self, iteration, x, y, color):
        c = color_to_letter((color, iteration))
        self.entries.append(f"[Iter{iteration}]   TRY: ({x},{y}) に色 {c} を配置")

    def skip(self, reason):
        self.entries.append(f"      SKIP: {reason}")

    def dump(self):
        print("\n".join(self.entries))
        self.entries.clear()

# --- FieldUtils クラス ---
class FieldUtils:
    @staticmethod
    def apply_gravity(field, columns, rows):
        """
        最上段(y=0)は固定して動かさない。
        y=1..rows-1 のみ重力適用（= 下から13段目は落下可能、最上段は落下しない）。
        """
        for x in range(columns):
            top_fixed = field[0][x]  # 最上段は保持
            stack = [field[y][x] for y in reversed(range(1, rows)) if field[y][x] is not None]
            for y in reversed(range(1, rows)):
                field[y][x] = stack.pop(0) if stack else None
            field[0][x] = top_fixed

    @staticmethod
    def get_connected_cells(field, x, y, columns, rows, directions):
        # 下から14段目(最上段:y=0)と13段目(y=1)は連結に参加しない
        if y <= 1:
            return [(x, y)]
        base_color = get_cell_color(field[y][x])
        visited = [[False] * columns for _ in range(rows)]
        visited[y][x] = True
        queue = deque([(x, y)])
        connected = [(x, y)]
        while queue:
            cx, cy = queue.popleft()
            for dx, dy in directions:
                nx, ny = cx + dx, cy + dy
                # 連結探索は y>=2 のみ対象
                if (0 <= nx < columns and 2 <= ny < rows and
                        not visited[ny][nx] and
                        field[ny][nx] is not None and
                        get_cell_color(field[ny][nx]) == base_color):
                    visited[ny][nx] = True
                    queue.append((nx, ny))
                    connected.append((nx, ny))
        return connected

    @staticmethod
    def extract_additions(merged, base, columns, rows):
        additions = []
        for x in range(columns):
            col_merged = [merged[y][x] for y in range(rows) if merged[y][x] is not None]
            col_base = [base[y][x] for y in range(rows) if base[y][x] is not None]
            diff = len(col_merged) - len(col_base)
            add = col_merged[:diff] if diff > 0 else []
            additions.append(add)
        return additions

    @staticmethod
    def add_accumulated(accum, new_add, columns):
        result = []
        for x in range(columns):
            result.append(new_add[x] + accum[x])
        return result

# --- 連鎖検出器クラス ---
class 連鎖検出器:
    # 4方向（上下左右）
    DIRECTIONS = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    
    def __init__(self, field, columns, rows, log_func=None, logger=None):
        self.field = copy.deepcopy(field)
        self.columns = columns
        self.rows = rows
        self.log = log_func if log_func is not None else (lambda msg: None)
        self.logger = logger or PlacementLogger()

    def _is_in_range(self, x, y):
        return 0 <= x < self.columns and 0 <= y < self.rows

    def _get_connected_cells(self, sx, sy):
        return FieldUtils.get_connected_cells(self.field, sx, sy, self.columns, self.rows, self.DIRECTIONS)

    def _get_connected_cells_in_field(self, field, sx, sy):
        return FieldUtils.get_connected_cells(field, sx, sy, self.columns, self.rows, self.DIRECTIONS)

    # ぷよA候補グループの検出と必要な追加配置の統合処理
    # ★ 修正：iteration パラメータを追加（初期配置の場合は 0、追加配置の場合は 1 以上）
    def check_and_place_puyos_for_color(self, x, y, base_color,
                                        blocked_columns=None,
                                        previous_additions=None,
                                        iteration=0):
        # ① TRY ログ
        self.logger.attempt(iteration, x, y, base_color)

        placed_positions = []

        # 範囲外 or 埋まっている
        if not self._is_in_range(x, y) or self.field[y][x] is not None:
            self.logger.skip("対象セルが範囲外または既に埋まっている")
            return False

        adjacent_groups = []
        for dx, dy in [(1, 0), (-1, 0), (0, 1)]:
            nx, ny = x + dx, y + dy
            # ★ y>=2 のみ隣接同色の対象
            if self._is_in_range(nx, ny) and ny >= 2 and self.field[ny][nx] is not None and get_cell_color(self.field[ny][nx]) == base_color:
                group = self._get_connected_cells(nx, ny)
                if not any(set(group) & set(existing) for existing in adjacent_groups):
                    adjacent_groups.append(group)
        if not adjacent_groups:
            self.logger.skip("隣接する同色ぷよがない")
            return False

        # ぷよAグループの作成
        puyoA_group = set()
        for group in adjacent_groups:
            puyoA_group.update(group)
        adjacency_base = set(puyoA_group)
        allowed_cols = set()
        for (cx, cy) in puyoA_group:
            allowed_cols.add(cx)
            if cx > 0:
                allowed_cols.add(cx - 1)
            if cx < self.columns - 1:
                allowed_cols.add(cx + 1)

        total_adjacent = sum(len(group) for group in adjacent_groups)
        effective_adjacent = total_adjacent if total_adjacent < 4 else 3
        needed = 4 - effective_adjacent  # 追加すべきぷよBの個数

        self.logger.entries.append(
            f"[Iter{iteration}] 候補: ({x},{y}) に色 {color_to_letter((base_color, iteration))}、追加B必要数={needed}"
        )

        # ★ ここで B#1～B#needed をループ
        for i in range(needed):
            self.logger.entries.append(f"  -- 追加配置 {i+1} 回目探索 --")
            best_chain = -1
            best_candidate_field = None
            best_details = None

            for col in sorted(allowed_cols):
                if blocked_columns and col in blocked_columns:
                    self.logger.skip(f"列 {col} はブロック済み")
                    continue

                candidate_y = self._find_bottom_empty_cell(self.field, col)
                if candidate_y is None:
                    self.logger.skip(f"列 {col} に空きセルなし")
                    continue

                candidate_field = copy.deepcopy(self.field)
                # ★ 追加配置の際はタプル (base_color, iteration) を設定
                candidate_field[candidate_y][col] = (base_color, iteration)
                FieldUtils.apply_gravity(candidate_field, self.columns, self.rows)
                
                final_y = None
                for yy in reversed(range(self.rows)):
                    if candidate_field[yy][col] is not None and self.field[yy][col] is None:
                        final_y = yy
                        break
                if final_y is None:
                    continue

                adjacent = False
                for dx, dy in self.DIRECTIONS:
                    nx, ny = col + dx, final_y + dy
                    if (nx, ny) in adjacency_base:
                        adjacent = True
                        break
                if not adjacent:
                    self.logger.skip(f"列 {col} 位置({col},{final_y}) がAグループと非隣接")
                    continue

                candidate_group = self._get_connected_cells_in_field(candidate_field, col, final_y)
                candidate_field_copy = copy.deepcopy(candidate_field)
                detector_candidate_copy = 連鎖検出器(candidate_field_copy, self.columns, self.rows)
                candidate_chain = detector_candidate_copy.simulate_chain()
                # 配置色をアルファベットで分かりやすく表示
                letter = color_to_letter((base_color, iteration))
                self.logger.entries.append(f"    CANDIDATE: 列 {col} 位置({col},{final_y}) → chain {candidate_chain}")
                if candidate_chain > best_chain:
                    best_chain = candidate_chain
                    best_candidate_field = candidate_field
                    best_details = (col, final_y)
                    # 更新ログにも同じ色情報を含める
                    self.logger.entries.append(f"【追加候補評価】更新: 列 {col} の位置 ({col}, {final_y}), 色 {letter} が最高連鎖数 {candidate_chain} を記録")
            if best_candidate_field is None:
                self.logger.skip(f"{i+1} 個目の追加候補が見つからず")
                return False
            else:
                col, final_y = best_details
                # B配置ログをインライン化（色表示を tuple 化して正しく取得）
                letter = color_to_letter((base_color, iteration))
                self.logger.entries.append(
                    f"    B#{i+1}: 列 {col} 行 {final_y} に色 {letter} を配置"
                )
                placed_positions.append((col, final_y))

                # フィールド適用 & 次の探索準備
                self.field = best_candidate_field
                adjacency_base.add((col, final_y))
                allowed_cols.add(col)
                if col > 0:
                    allowed_cols.add(col - 1)
                if col < self.columns - 1:
                    allowed_cols.add(col + 1)

        # ★ 3 回目まで回ったらここでまとめ & return（ループをリセットしない）
        cols = ",".join(str(col) for col, _ in placed_positions)
        self.logger.entries.append(
            f"    配置合計: {len(placed_positions)} 個, 列一覧: {cols}"
        )
        return True

    def simulate_chain(self):
        chain_count = 0
        while True:
            groups = self._find_4_or_more()
            if not groups:
                break
            # 1連鎖目のみで異なる iterationID の同時消しをチェック
            if chain_count == 0:
                iteration_ids = set()
                for group in groups:
                    for (gx, gy) in group:
                        cell = self.field[gy][gx]
                        if cell is not None:
                            iter_id = cell[1] if isinstance(cell, tuple) else 0
                            if iter_id != 0:
                                iteration_ids.add(iter_id)
                if len(iteration_ids) > 1:
                    self.logger.entries.append("【同時消し検出】1連鎖目で異なる追加配置 (IDs={}) が消える".format(iteration_ids))
                    return -1
            chain_count += 1
            for grp in groups:
                for (gx, gy) in grp:
                    self.field[gy][gx] = None
            FieldUtils.apply_gravity(self.field, self.columns, self.rows)
        return chain_count

    def reflect_to(self, external_field):
        for y in range(self.rows):
            for x in range(self.columns):
                external_field[y][x] = self.field[y][x]

    def simulate_chain_with_mapping(self):
        self.simulate_chain()
        return copy.deepcopy(self.field), None

    def _find_bottom_empty_cell(self, field, col):
        for y in reversed(range(self.rows)):
            if field[y][col] is None:
                return y
        return None

    def _find_4_or_more(self):
        visited = [[False] * self.columns for _ in range(self.rows)]
        found_groups = []
        for y in range(2, self.rows):  # ★ y=2(下から12段目)以降のみ探索
            for x in range(self.columns):
                if self.field[y][x] is None or visited[y][x]:
                    continue
                group = self._get_connected_cells(x, y)
                for gx, gy in group:
                    visited[gy][gx] = True
                if len(group) >= 4:
                    found_groups.append(group)
        return found_groups

# --- 連鎖生成器クラス ---
class 連鎖生成器:
    def __init__(self, original_field, columns, rows, log_func=None, allow_full_column=False):
        self.original_field = original_field
        self.columns = columns
        self.rows = rows
        self.log = log_func if log_func is not None else (lambda msg: None)
        self.allow_full_column = allow_full_column

    # ★ 修正：find_best_arrangement にも iteration パラメータを追加
    def find_best_arrangement(self, target_color=None, blocked_columns=None, preferred_columns=None, previous_additions=None, iteration=0):
        candidates = []
        candidate_columns = []
        if preferred_columns:
            candidate_columns.extend(sorted(list(preferred_columns)))
        for x in range(self.columns):
            if blocked_columns and x in blocked_columns:
                continue
            if preferred_columns and x in preferred_columns:
                continue
            candidate_columns.append(x)

        for x in candidate_columns:
            bottom_y = self._find_bottom_empty_cell_in_column(x)
            if bottom_y is None:
                continue

            if target_color is not None:
                colors_to_try = [target_color]
            else:
                colors_to_try = self._get_neighbor_colors(x, bottom_y)
                if not colors_to_try:
                    continue

            for color in colors_to_try:
                # ―― ① ベース候補見出しログ ――
                detector = 連鎖検出器(copy.deepcopy(self.original_field),
                                   self.columns, self.rows,
                                   log_func=print,
                                   logger=PlacementLogger())
                detector.logger.header(iteration, x, bottom_y, color)

                # ―― ② 実際の配置試行 ――
                test_field = copy.deepcopy(self.original_field)
                placed_ok = detector.check_and_place_puyos_for_color(
                    x, bottom_y, color,
                    blocked_columns=blocked_columns,
                    previous_additions=previous_additions,
                    iteration=iteration
                )
                # 各候補評価後にログを出力
                detector.logger.dump()
                if not placed_ok:
                    continue

                detector.reflect_to(test_field)
                chain_count = detector.simulate_chain()
                print(f"【候補評価】列{x} 位置({x},{bottom_y}) 色 {color_to_letter((color,iteration))} → chain_count={chain_count}")
                if chain_count > 0:
                    candidates.append((chain_count, test_field, (x, bottom_y)))
        candidates.sort(key=lambda tup: tup[0], reverse=True)
        return candidates

    def _find_bottom_empty_cell_in_column(self, col):
        for y in reversed(range(self.rows)):
            if self.original_field[y][col] is None:
                return y
        if self.allow_full_column:
            return 0
        return None

    def _get_neighbor_colors(self, x, y):
        directions = [(1, 0), (-1, 0), (0, 1)]
        colors = set()
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < self.columns and 0 <= ny < self.rows:
                c = self.original_field[ny][nx]
                if c is not None:
                    colors.add(get_cell_color(c))
        return list(colors)

# --- Puyo クラス ---
class Puyo:
    def __init__(self, color, x, y):
        self.color = color
        self.x = x
        self.y = y
        self.fixed = False

    def draw(self, surface, offset_x=0, offset_y=0):
        pygame.draw.rect(surface, self.color,
                         (offset_x + self.x * CELL_SIZE, offset_y + self.y * CELL_SIZE, CELL_SIZE, CELL_SIZE))
        pygame.draw.rect(surface, (255, 255, 255),
                         (offset_x + self.x * CELL_SIZE, offset_y + self.y * CELL_SIZE, CELL_SIZE, CELL_SIZE), 1)

# --- PuyoPair クラス ---
class PuyoPair:
    REL = [(0, 1), (1, 0), (0, -1), (-1, 0)]
    def __init__(self, c1, c2):
        self.axis = Puyo(c1, 2, 1)
        self.rot = 2
        self.child = self._update_child(c2)

    def _update_child(self, c2):
        dx, dy = self.REL[self.rot]
        return Puyo(c2, self.axis.x + dx, self.axis.y + dy)

    def is_vertical(self):
        return self.axis.x == self.child.x

    def move_horizontal(self, dx):
        nax = self.axis.x + dx
        ncx = self.child.x + dx
        if (0 <= nax < COLUMNS and field[self.axis.y][nax] is None) and (0 <= ncx < COLUMNS and field[self.child.y][ncx] is None):
            self.axis.x = nax
            self.child.x = ncx

    def rotate(self, direction):
        if direction == 'clockwise':
            nr = (self.rot + 1) % 4
        else:
            nr = (self.rot - 1) % 4
        dx, dy = self.REL[nr]
        ncx = self.axis.x + dx
        ncy = self.axis.y + dy
        old_x, old_y = self.axis.x, self.axis.y

        if 0 <= ncx < COLUMNS and 0 <= ncy < ROWS and field[ncy][ncx] is None:
            self.rot = nr
            self.child.x = ncx
            self.child.y = ncy
        else:
            shift = 0
            if ncx < 0:
                shift = 1
            elif ncx >= COLUMNS:
                shift = -1
            if shift:
                test_ax = self.axis.x + shift
                if 0 <= test_ax < COLUMNS and field[self.axis.y][test_ax] is None:
                    self.axis.x = test_ax
                    ncx = self.axis.x + dx
                    ncy = self.axis.y + dy
                    if 0 <= ncx < COLUMNS and 0 <= ncy < ROWS and field[ncy][ncx] is None:
                        self.rot = nr
                        self.child.x = ncx
                        self.child.y = ncy
                    else:
                        self.axis.x = old_x
            else:
                upy = self.axis.y - 1
                if 0 <= upy < ROWS and field[upy][self.axis.x] is None:
                    self.axis.y = upy
                    ncx = self.axis.x + dx
                    ncy = self.axis.y + dy
                    if 0 <= ncx < COLUMNS and 0 <= ncy < ROWS and field[ncy][ncx] is None:
                        self.rot = nr
                        self.child.x = ncx
                        self.child.y = ncy
                    else:
                        self.axis.y = old_y

    def drop_once(self):
        if self.is_vertical():
            if self.axis.y > self.child.y:
                d, u = self.axis, self.child
            else:
                d, u = self.child, self.axis
            ny = d.y + 1
            if ny < ROWS and field[ny][d.x] is None:
                d.y += 1
                u.y += 1
            else:
                self._fix()
        else:
            ax, ay = self.axis.x, self.axis.y + 1
            cx, cy = self.child.x, self.child.y + 1
            can_axis_fall = (ay < ROWS and field[ay][ax] is None)
            can_child_fall = (cy < ROWS and field[cy][cx] is None)
            if can_axis_fall and can_child_fall:
                self.axis.y += 1
                self.child.y += 1
            elif can_axis_fall:
                self.axis.y += 1
            elif can_child_fall:
                self.child.y += 1
            else:
                self._fix()

    def hard_drop(self):
        while not (self.axis.fixed and self.child.fixed):
            self.drop_once()

    def _fix(self):
        # ★ 固定時は iterationID=0（初期盤面扱い）で固定する
        if not self.axis.fixed:
            field[self.axis.y][self.axis.x] = (self.axis.color, 0)
            self.axis.fixed = True
            if self.axis.y == 0:
                field[self.axis.y][self.axis.x] = None
        if not self.child.fixed:
            field[self.child.y][self.child.x] = (self.child.color, 0)
            self.child.fixed = True
            if self.child.y == 0:
                field[self.child.y][self.child.x] = None

    def draw(self, surface):
        self.axis.draw(surface)
        self.child.draw(surface)

def create_new_puyopair():
    c1 = random.choice(PUYO_COLORS)
    c2 = random.choice(PUYO_COLORS)
    return PuyoPair(c1, c2)

def generate_hand():
    return [create_new_puyopair() for _ in range(64)]

hand = []       
hand_index = 0  
current_pair = None
next_pair = None
double_next_pair = None

def reset_game():
    global field, current_pair, next_pair, double_next_pair, hand, hand_index
    field[:] = [[None for _ in range(COLUMNS)] for _ in range(ROWS)]
    hand[:] = generate_hand()
    hand_index = 3
    current_pair = hand[0]
    next_pair = hand[1]
    double_next_pair = hand[2]
    undo_stack.clear()

def save_state():
    st = {
        'field': copy.deepcopy(field),
        'cpair': copy.deepcopy(current_pair),
        'n1': copy.deepcopy(next_pair),
        'n2': copy.deepcopy(double_next_pair),
        'hand_index': hand_index,
        'hand': copy.deepcopy(hand)
    }
    undo_stack.append(st)

def restore_state():
    global field, current_pair, next_pair, double_next_pair, hand_index, hand
    if undo_stack:
        st = undo_stack.pop()
        field[:] = copy.deepcopy(st['field'])
        current_pair = copy.deepcopy(st['cpair'])
        next_pair = copy.deepcopy(st['n1'])
        double_next_pair = copy.deepcopy(st['n2'])
        hand_index = st['hand_index']
        hand = copy.deepcopy(st['hand'])
    else:
        print("No undo state left.")

# --- 4つ以上の連結グループ検出（ぷよセルの比較に get_cell_color を利用） ---
def find_4plus_groups():
    visited = [[False]*COLUMNS for _ in range(ROWS)]
    result = []
    for y in range(2, ROWS):  # ★ y=2(=下から12段目)以降のみ探索
        for x in range(COLUMNS):
            if field[y][x] is None or visited[y][x]:
                continue
            base_color = get_cell_color(field[y][x])
            q = deque([(x, y)])
            visited[y][x] = True
            group = [(x, y)]
            while q:
                cx, cy = q.popleft()
                for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                    nx, ny = cx + dx, cy + dy
                    if 0 <= nx < COLUMNS and 2 <= ny < ROWS and not visited[ny][nx] and field[ny][nx] is not None and get_cell_color(field[ny][nx]) == base_color:
                        visited[ny][nx] = True
                        q.append((nx, ny))
                        group.append((nx, ny))
            if len(group) >= 4:
                result.append(group)
    return result

def remove_puyos(groups):
    for grp in groups:
        for (x, y) in grp:
            field[y][x] = None

def apply_gravity():
    FieldUtils.apply_gravity(field, COLUMNS, ROWS)

def extract_additions(merged, base):
    return FieldUtils.extract_additions(merged, base, COLUMNS, ROWS)

def add_accumulated(accum, new_add):
    return FieldUtils.add_accumulated(accum, new_add, COLUMNS)

def build_field_from_accum(original, accum):
    merged = [[None for _ in range(COLUMNS)] for _ in range(ROWS)]
    for x in range(COLUMNS):
        col_base = [original[y][x] for y in range(ROWS) if original[y][x] is not None]
        new_col = accum[x] + col_base
        for y in range(ROWS):
            idx = len(new_col) - (ROWS - y)
            merged[y][x] = new_col[idx] if idx >= 0 else None
    FieldUtils.apply_gravity(merged, COLUMNS, ROWS)
    return merged

# --- 追加配置候補の再試行用関数 ---
def try_extended_cleanup_arrangement(pre_chain_field, original_field, base_chain, iteration, blocked_columns=None, previous_additions=None):
    detector = 連鎖検出器(copy.deepcopy(pre_chain_field), COLUMNS, ROWS)
    leftover_field, _ = detector.simulate_chain_with_mapping()

    generator = 連鎖生成器(leftover_field, COLUMNS, ROWS, allow_full_column=True)

    if iteration == 1:
        base_for_seed = original_field
    else:
        base_for_seed = leftover_field

    candidate_groups = []
    directions = [(1, 0), (-1, 0), (0, 1)]
    for col in range(COLUMNS):
        empty_y = None
        for y in reversed(range(ROWS)):
            if base_for_seed[y][col] is None:
                empty_y = y
                break
        if empty_y is None:
            continue
        neighbor_positions = []
        for dx, dy in directions:
            nx = col + dx
            ny = empty_y + dy
            if 0 <= nx < COLUMNS and 0 <= ny < ROWS and base_for_seed[ny][nx] is not None:
                neighbor_positions.append((nx, ny))
        for nx, ny in neighbor_positions:
            candidate_color = get_cell_color(base_for_seed[ny][nx])
            group = set(FieldUtils.get_connected_cells(base_for_seed, nx, ny, COLUMNS, ROWS, 連鎖検出器.DIRECTIONS))
            for dx, dy in directions:
                ax = col + dx
                ay = empty_y + dy
                if 0 <= ax < COLUMNS and 0 <= ay < ROWS and base_for_seed[ay][ax] is not None and get_cell_color(base_for_seed[ay][ax]) == candidate_color:
                    extra_group = set(FieldUtils.get_connected_cells(base_for_seed, ax, ay, COLUMNS, ROWS, 連鎖検出器.DIRECTIONS))
                    group = group.union(extra_group)
            candidate_field = copy.deepcopy(base_for_seed)
            # ★ 追加配置の場合、iterationID を仮に target として配置（ここでは candidate_color の追加配置）
            candidate_field[empty_y][col] = (candidate_color, iteration)
            FieldUtils.apply_gravity(candidate_field, COLUMNS, ROWS)
            detector_candidate = 連鎖検出器(candidate_field, COLUMNS, ROWS)
            chain_count = detector_candidate.simulate_chain()
            candidate_groups.append((chain_count, group, col, candidate_color, (col, empty_y)))
    if candidate_groups:
        best_candidate = max(candidate_groups, key=lambda x: x[0])
        best_chain_value, best_group, best_col, best_color, best_empty = best_candidate
        target_color = best_color
        puyoA_group = best_group
    else:
        print("【警告】イテレーション {}: 有効なぷよA候補が見つかりません。".format(iteration))
        puyoA_group = set()
        return base_chain, None

    candidates = generator.find_best_arrangement(
        target_color=target_color,
        blocked_columns=blocked_columns,
        previous_additions=previous_additions,
        iteration=iteration  # ★ ここで iteration を渡す
    )
    ext_chain = None
    ext_field = None
    merged_field_final = None
    for candidate in candidates:
        candidate_chain, candidate_field, candidate_coords = candidate
        simulate_test = copy.deepcopy(candidate_field)
        detector_ext = 連鎖検出器(simulate_test, COLUMNS, ROWS)
        new_chain_candidate = detector_ext.simulate_chain()

        additional_counts = [0] * COLUMNS
        for x in range(COLUMNS):
            count_leftover = sum(1 for y in range(ROWS) if leftover_field[y][x] is not None)
            count_candidate = sum(1 for y in range(ROWS) if candidate_field[y][x] is not None)
            additional_counts[x] = max(0, count_candidate - count_leftover)
        if sum(additional_counts) == 0:
            continue

        merged_field_candidate = copy.deepcopy(pre_chain_field)
        for x in range(COLUMNS):
            col_base = [merged_field_candidate[y][x] for y in range(ROWS) if merged_field_candidate[y][x] is not None]
            candidate_column = [candidate_field[y][x] for y in range(ROWS) if candidate_field[y][x] is not None]
            additional = candidate_column[:additional_counts[x]]
            new_col = additional + col_base
            for y in range(ROWS):
                idx = len(new_col) - (ROWS - y)
                merged_field_candidate[y][x] = new_col[idx] if idx >= 0 else None
        FieldUtils.apply_gravity(merged_field_candidate, COLUMNS, ROWS)

        # ★ マージ後のフィールドにも同時消しチェック
        merged_detector = 連鎖検出器(copy.deepcopy(merged_field_candidate), COLUMNS, ROWS)
        merged_chain = merged_detector.simulate_chain()
        if merged_chain == -1:
            print(f"【警告】イテレーション {iteration}: マージ後に異なる追加配置が同時消しとなるため、この候補をスキップします。")
            continue

        current_new_additions_candidate = set()
        for y in range(ROWS):
            for x in range(COLUMNS):
                if merged_field_candidate[y][x] is not None and pre_chain_field[y][x] is None:
                    current_new_additions_candidate.add((x, y))
                    # ※ 以下、候補マージ後の色を target_color に統一しておく（描画時は get_cell_color で色を取得）
                    merged_field_candidate[y][x] = (target_color, iteration)

        puyoC_positions_candidate = set()
        for y in range(ROWS):
            for x in range(COLUMNS):
                if merged_field_candidate[y][x] is not None and get_cell_color(merged_field_candidate[y][x]) == target_color and (x, y) not in puyoA_group:
                    puyoC_positions_candidate.add((x, y))
        puyoC_positions_candidate -= current_new_additions_candidate

        conflict = False
        for (bx, by) in current_new_additions_candidate:
            for dx, dy in 連鎖検出器.DIRECTIONS:
                nx, ny = bx + dx, by + dy
                if 0 <= nx < COLUMNS and 0 <= ny < ROWS:
                    if (nx, ny) in puyoC_positions_candidate:
                        conflict = True
                        break
            if conflict:
                break

        if conflict:
            print("【警告】候補配置（候補チェーン数: {}）はぷよCとの隣接で不適切なため、次の候補を試みます。".format(candidate_chain))
            continue
        else:
            ext_chain = new_chain_candidate
            ext_field = candidate_field
            merged_field_final = merged_field_candidate
            break

    if ext_field is None:
        return base_chain, None

    return ext_chain, merged_field_final

# --- ビームサーチ版 iterative_chain_clearing ---
def iterative_chain_clearing(original_field, base_chain=None, beam_width=3, max_depth=3):
    BeamState = namedtuple(
        'BeamState',
        ['field', 'acc_chain', 'acc_adds', 'blocked_cols', 'prev_adds', 'iteration']
    )

    # 初期連鎖数を取得
    detector0 = 連鎖検出器(copy.deepcopy(original_field), COLUMNS, ROWS)
    baseline_chain = base_chain if base_chain is not None else detector0.simulate_chain()

    # 初期ビーム状態
    initial = BeamState(
        field=copy.deepcopy(original_field),
        acc_chain=baseline_chain,
        acc_adds=[[] for _ in range(COLUMNS)],
        blocked_cols=set(),
        prev_adds={},
        iteration=0
    )
    beam = [initial]
    best_state = initial

    # ビームサーチループ
    while beam:
        # 全ての枝が深さ制限に達したら終了
        if all(s.iteration >= max_depth for s in beam):
            break
        next_beam = []
        any_extended = False

        for state in beam:
            # 深さ制限：この枝はもう展開せず、そのまま次ビームに保持
            if state.iteration >= max_depth:
                next_beam.append(state)
                continue
            it = state.iteration + 1
            new_chain, candidate = try_extended_cleanup_arrangement(
                state.field,
                original_field,
                baseline_chain,
                it,
                blocked_columns=(state.blocked_cols if it > 1 else None),
                previous_additions=state.prev_adds
            )
            if candidate is None:
                next_beam.append(state)
            else:
                any_extended = True
                # 追加分を計算 & 累積
                new_add = extract_additions(candidate, state.field)
                accum_adds = FieldUtils.add_accumulated(state.acc_adds, new_add, COLUMNS)

                # blocked_columns 更新（1回目のみ）
                blk = set(state.blocked_cols)
                if it == 1:
                    for col_idx, adds in enumerate(new_add):
                        if adds:
                            blk.add(col_idx)

                # previous_additions 更新
                new_prev = {}
                for y in range(ROWS):
                    for x in range(COLUMNS):
                        if candidate[y][x] is not None and state.field[y][x] is None:
                            new_prev[(x, y)] = candidate[y][x]
                merged_prev = dict(state.prev_adds)
                merged_prev.update(new_prev)

                # 次ビーム状態を生成
                next_field = build_field_from_accum(original_field, accum_adds)
                next_beam.append(BeamState(
                    field=next_field,
                    acc_chain=state.acc_chain + new_chain,
                    acc_adds=accum_adds,
                    blocked_cols=blk,
                    prev_adds=merged_prev,
                    iteration=it
                ))

        # 追加配置が一度も発生しなかったら探索打ち切り
        if not any_extended:
            break
        # ビーム幅でソート＆プルーニング
        beam = sorted(next_beam, key=lambda s: s.acc_chain, reverse=True)[:beam_width]
        # 全体最良状態を更新
        best_state = max(best_state, *beam, key=lambda s: s.acc_chain)

    return best_state.acc_chain, best_state.field

def update_right_side_preview():
    global best_arrangement_chain, best_arrangement_field
    pre_chain_field = copy.deepcopy(field)
    total_chain, final_field = iterative_chain_clearing(pre_chain_field, None)
    if final_field is not None:
        best_arrangement_chain = total_chain
        best_arrangement_field = copy.deepcopy(final_field)
        print_board(pre_chain_field, "追加配置対象盤面（更新時）")
        print_board(best_arrangement_field, "右側フィールド盤面")
    else:
        best_arrangement_chain = 0
        best_arrangement_field = None

best_arrangement_chain = 0
best_arrangement_field = None

def draw_grid(surface, offset_x=0, offset_y=0):
    for cx in range(COLUMNS + 1):
        px = offset_x + cx * CELL_SIZE
        pygame.draw.line(surface, (40, 40, 40), (px, offset_y), (px, offset_y + CELL_SIZE * ROWS), 1)
    for cy in range(ROWS + 1):
        py = offset_y + cy * CELL_SIZE
        pygame.draw.line(surface, (40, 40, 40), (offset_x, py), (offset_x + CELL_SIZE * COLUMNS, py), 1)

def draw_next(surface, n1, n2, font):
    tx = font.render("Next:", True, (255, 255, 255))
    surface.blit(tx, (COLUMNS * CELL_SIZE + 20, 20))
    if n1:
        bx = COLUMNS * CELL_SIZE + 20
        by = 60
        pygame.draw.rect(surface, n1.axis.color, (bx, by, CELL_SIZE, CELL_SIZE))
        pygame.draw.rect(surface, (255, 255, 255), (bx, by, CELL_SIZE, CELL_SIZE), 1)
        dx, dy = n1.REL[n1.rot]
        cx = bx + dx * CELL_SIZE
        cy = by + dy * CELL_SIZE
        pygame.draw.rect(surface, n1.child.color, (cx, cy, CELL_SIZE, CELL_SIZE))
        pygame.draw.rect(surface, (255, 255, 255), (cx, cy, CELL_SIZE, CELL_SIZE), 1)

    tx2 = font.render("Double:", True, (255, 255, 255))
    surface.blit(tx2, (COLUMNS * CELL_SIZE + 20, 130))
    if n2:
        bx = COLUMNS * CELL_SIZE + 20
        by = 170
        pygame.draw.rect(surface, n2.axis.color, (bx, by, CELL_SIZE, CELL_SIZE))
        pygame.draw.rect(surface, (255, 255, 255), (bx, by, CELL_SIZE, CELL_SIZE), 1)
        dx, dy = n2.REL[n2.rot]
        cx = bx + dx * CELL_SIZE
        cy = by + dy * CELL_SIZE
        pygame.draw.rect(surface, n2.child.color, (cx, cy, CELL_SIZE, CELL_SIZE))
        pygame.draw.rect(surface, (255, 255, 255), (cx, cy, CELL_SIZE, CELL_SIZE), 1)

pygame.init()
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("ぷよぷよ(サンプル)")
clock = pygame.time.Clock()
font = pygame.font.SysFont(None, 36)

reset_game()

while True:
    dt = clock.tick(60)
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()
        elif event.type == pygame.KEYDOWN:
            # リセットと左右移動・回転は KEYDOWN で
            if event.key == pygame.K_r:
                reset_game()
                update_right_side_preview()
            elif not (current_pair.axis.fixed and current_pair.child.fixed):
                if event.key == pygame.K_LEFT:
                    current_pair.move_horizontal(-1)
                elif event.key == pygame.K_RIGHT:
                    current_pair.move_horizontal(1)
                elif event.key == pygame.K_z:
                    current_pair.rotate('clockwise')
                elif event.key == pygame.K_x:
                    current_pair.rotate('counterclockwise')
        elif event.type == pygame.KEYUP:
            # ハードドロップとアンドゥは KEYUP で一度だけ処理
            if event.key == pygame.K_DOWN:
                save_state()
                current_pair.hard_drop()
            elif event.key == pygame.K_UP:
                restore_state()
                update_right_side_preview()

    if current_pair.axis.fixed and current_pair.child.fixed:
        # 既存のゲームオーバー条件（出現位置の詰まり）
        if field[1][2] is not None or field[0][2] is not None:
            print("Game Over!")
            pygame.quit()
            sys.exit()

        chain_count = 0
        groups = find_4plus_groups()
        while groups:
            chain_count += 1
            remove_puyos(groups)
            screen.fill((0, 0, 0))
            for ry in range(ROWS):
                for rx in range(COLUMNS):
                    c = field[ry][rx]
                    if c:
                        # ★ 描画時はタプルの場合、get_cell_colorで色取得
                        pygame.draw.rect(screen, get_cell_color(c), (rx * CELL_SIZE, ry * CELL_SIZE, CELL_SIZE, CELL_SIZE))
            draw_grid(screen)
            pygame.display.flip()
            pygame.time.delay(300)

            apply_gravity()
            screen.fill((0, 0, 0))
            for ry in range(ROWS):
                for rx in range(COLUMNS):
                    c = field[ry][rx]
                    if c:
                        pygame.draw.rect(screen, get_cell_color(c), (rx * CELL_SIZE, ry * CELL_SIZE, CELL_SIZE, CELL_SIZE))
            draw_grid(screen)
            pygame.display.flip()
            pygame.time.delay(300)
            groups = find_4plus_groups()

        # ★ 連鎖・重力の解決後に「左から3列目×下から12段目」が埋まっていたらゲームオーバー
        # 左から3列目 → x=2, 下から12段目 → y=2
        if field[2][2] is not None:
            print("Game Over!（左から3列目の12段目が埋まっています）")
            pygame.quit()
            sys.exit()

        current_pair = next_pair
        next_pair = double_next_pair
        double_next_pair = hand[hand_index]
        hand_index = (hand_index + 1) % 64
        update_right_side_preview()

    screen.fill((0, 0, 0))
    for ry in range(ROWS):
        for rx in range(COLUMNS):
            c = field[ry][rx]
            if c:
                pygame.draw.rect(screen, get_cell_color(c), (rx * CELL_SIZE, ry * CELL_SIZE, CELL_SIZE, CELL_SIZE))

    draw_grid(screen, 0, 0)

    if not (current_pair.axis.fixed and current_pair.child.fixed):
        current_pair.draw(screen)

    draw_next(screen, next_pair, double_next_pair, font)

    if best_arrangement_field:
        for ry in range(ROWS):
            for rx in range(COLUMNS):
                c = best_arrangement_field[ry][rx]
                if c:
                    pygame.draw.rect(screen, get_cell_color(c), (BEST_FIELD_OFFSET_X + rx * CELL_SIZE, ry * CELL_SIZE, CELL_SIZE, CELL_SIZE))
        draw_grid(screen, BEST_FIELD_OFFSET_X, 0)
        txt = font.render(f"MaxChain: {best_arrangement_chain}", True, (255, 255, 255))
        screen.blit(txt, (BEST_FIELD_OFFSET_X + 20, 300))
    else:
        draw_grid(screen, BEST_FIELD_OFFSET_X, 0)

    pygame.display.flip()
