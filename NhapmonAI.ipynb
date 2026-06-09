# BƯỚC 1: CÀI ĐẶT THƯ VIỆN CẦN THIẾT
!pip install gradio pillow numpy matplotlib

import io
import math
import random
import time
from collections import deque
import heapq
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw
import gradio as gr

matplotlib.use('Agg')


# BƯỚC 2: CÁC HÀM ĐÁNH GIÁ HEURISTIC TÙY CHỌN

def get_manhattan_distance(state, size):
    dist = 0
    for i, value in enumerate(state):
        if value == 0:
            continue
        goal_idx = value - 1
        x1, y1 = i // size, i % size
        x2, y2 = goal_idx // size, goal_idx % size
        dist += abs(x1 - x2) + abs(y1 - y2)
    return dist


def get_euclidean_distance(state, size):
    dist = 0.0
    for i, value in enumerate(state):
        if value == 0:
            continue
        goal_idx = value - 1
        x1, y1 = i // size, i % size
        x2, y2 = goal_idx // size, goal_idx % size

        dist += math.isqrt((x1 - x2)**2 + (y1 - y2)**2)
    return int(dist)

def get_linear_conflict(state, size):
    conflict = get_manhattan_distance(state, size)
    for i in range(size):
        row = state[i * size : (i + 1) * size]
        for a in range(size):
            for b in range(a + 1, size):
                v1, v2 = row[a], row[b]
                if v1 != 0 and v2 != 0 and (v1 - 1) // size == i and (v2 - 1) // size == i and v1 > v2:
                    conflict += 2

    for j in range(size):
        col = [state[it * size + j] for it in range(size)]
        for a in range(size):
            for b in range(a + 1, size):
                v1, v2 = col[a], col[b]
                if v1 != 0 and v2 != 0 and (v1 - 1) % size == j and (v2 - 1) % size == j and v1 > v2:
                    conflict += 2
    return conflict

def get_misplaced_tiles(state, size):
    count = 0
    goal = list(range(1, size * size)) + [0]
    for i in range(len(state)):
        if state[i] != 0 and state[i] != goal[i]:
            count += 1
    return count

def compute_heuristic(state, size, h_type):
    if h_type == "Manhattan":
        return get_manhattan_distance(state, size)
    elif h_type == "Euclidean":
        return get_euclidean_distance(state, size)
    elif h_type == "Linear Conflict":
        return get_linear_conflict(state, size)
    elif h_type == "Misplaced Tiles":
        return get_misplaced_tiles(state, size)
    return 0


# BƯỚC 3: CÁC THUẬT TOÁN TÌM KIẾM

def solve_a_star_with_tree(start_state, size, h_type):
    goal_state = tuple(list(range(1, size * size)) + [0])
    start_tuple = tuple(start_state)
    tree_edges = []

    if start_tuple == goal_state:
        return [start_tuple], 0, 0.0, tree_edges

    start_time = time.time()
    h_start = compute_heuristic(start_tuple, size, h_type)
    open_set = [(h_start, 0, start_tuple, [start_tuple])]
    closed_set = set()
    node_count = 0

    while open_set:
        f, g, current, path = heapq.heappop(open_set)
        node_count += 1

        if current == goal_state:
            duration = round((time.time() - start_time) * 1000, 2)
            return path, node_count, duration, tree_edges

        if current in closed_set:
            continue
        closed_set.add(current)

        blank_idx = current.index(0)
        r, c = blank_idx // size, blank_idx % size
        directions = []
        if r > 0: directions.append(-size)
        if r < size - 1: directions.append(size)
        if c > 0: directions.append(-1)
        if c < size - 1: directions.append(1)

        for step in directions:
            neighbor_idx = blank_idx + step
            new_state = list(current)
            new_state[blank_idx], new_state[neighbor_idx] = new_state[neighbor_idx], new_state[blank_idx]
            new_state_tuple = tuple(new_state)

            if new_state_tuple not in closed_set:
                new_g = g + 1
                new_h = compute_heuristic(new_state_tuple, size, h_type)
                new_f = new_g + new_h

                if len(tree_edges) < 11:
                    tree_edges.append((current, new_state_tuple, new_f, new_g, new_h))

                heapq.heappush(open_set, (new_f, new_g, new_state_tuple, path + [new_state_tuple]))

    return [], node_count, round((time.time() - start_time) * 1000, 2), tree_edges


def solve_ida_star_with_tree(start_state, size, h_type, max_nodes=50000):
    goal_state = tuple(list(range(1, size * size)) + [0])
    start_tuple = tuple(start_state)
    tree_edges = []

    if start_tuple == goal_state:
        return [start_tuple], 0, 0.0, tree_edges

    start_time = time.time()
    node_count = 0

    def search(path, g, bound):
        nonlocal node_count
        node_count += 1
        current = path[-1]
        h = compute_heuristic(current, size, h_type)
        f = g + h

        if f > bound:
            return f, None
        if current == goal_state:
            return "FOUND", path
        if node_count > max_nodes:
            return "TIMEOUT", None

        min_val = float('inf')
        blank_idx = current.index(0)
        r, c = blank_idx // size, blank_idx % size

        directions = []
        if r > 0: directions.append(-size)
        if r < size - 1: directions.append(size)
        if c > 0: directions.append(-1)
        if c < size - 1: directions.append(1)

        for step in directions:
            neighbor_idx = blank_idx + step
            new_state = list(current)
            new_state[blank_idx], new_state[neighbor_idx] = new_state[neighbor_idx], new_state[blank_idx]
            new_state_tuple = tuple(new_state)

            if new_state_tuple not in path:
                if len(tree_edges) < 11:
                    new_h = compute_heuristic(new_state_tuple, size, h_type)
                    tree_edges.append((current, new_state_tuple, g + 1 + new_h, g + 1, new_h))

                path.append(new_state_tuple)
                t, res_path = search(path, g + 1, bound)
                if t == "FOUND":
                    return "FOUND", res_path
                if t == "TIMEOUT":
                    return "TIMEOUT", None
                if t < min_val:
                    min_val = t
                path.pop()
        return min_val, None

    bound = compute_heuristic(start_tuple, size, h_type)
    path = [start_tuple]

    while True:
        t, res_path = search(path, 0, bound)
        if t == "FOUND":
            duration = round((time.time() - start_time) * 1000, 2)
            return res_path, node_count, duration, tree_edges
        if t == "TIMEOUT" or t == float('inf'):
            duration = round((time.time() - start_time) * 1000, 2)
            return None, node_count, duration, tree_edges
        bound = t


def solve_bfs_with_tree(start_state, size, max_nodes=30000):
    goal_state = tuple(list(range(1, size * size)) + [0])
    start_tuple = tuple(start_state)
    tree_edges = []

    if start_tuple == goal_state:
        return [start_tuple], 0, 0.0, tree_edges

    start_time = time.time()
    queue = deque([(start_tuple, [start_tuple])])
    closed_set = set([start_tuple])
    node_count = 0

    while queue:
        current, path = queue.popleft()
        node_count += 1

        if current == goal_state:
            duration = round((time.time() - start_time) * 1000, 2)
            return path, node_count, duration, tree_edges

        if node_count > max_nodes:
            duration = round((time.time() - start_time) * 1000, 2)
            return None, node_count, duration, tree_edges

        blank_idx = current.index(0)
        r, c = blank_idx // size, blank_idx % size
        directions = []
        if r > 0: directions.append(-size)
        if r < size - 1: directions.append(size)
        if c > 0: directions.append(-1)
        if c < size - 1: directions.append(1)

        for step in directions:
            neighbor_idx = blank_idx + step
            new_state = list(current)
            new_state[blank_idx], new_state[neighbor_idx] = new_state[neighbor_idx], new_state[blank_idx]
            new_state_tuple = tuple(new_state)

            if new_state_tuple not in closed_set:
                closed_set.add(new_state_tuple)
                if len(tree_edges) < 11:
                    g_val = len(path)
                    tree_edges.append((current, new_state_tuple, g_val, g_val, 0))
                queue.append((new_state_tuple, path + [new_state_tuple]))

    return [], node_count, round((time.time() - start_time) * 1000, 2), tree_edges



# BƯỚC 4: VẼ ĐỒ THỊ VÀ BIỂU ĐỒ SO SÁNH


def draw_visual_search_tree(tree_edges, title_name, size=3, h_type="Manhattan", is_bfs=False):
    if not tree_edges:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.text(0.5, 0.5, f"Chưa có dữ liệu cây\ncủa {title_name}.",
                fontsize=11, ha='center', va='center', color='gray', style='italic')
        ax.axis('off')
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=130)
        plt.close(fig)
        buf.seek(0)
        return Image.open(buf)

    unique_nodes = set()
    node_data = {}
    root = tree_edges[0][0]

    if is_bfs:
        node_data[root] = {"g": 0, "h": 0}
    else:
        node_data[root] = {"g": 0, "h": compute_heuristic(root, size, h_type)}

    for parent, child, f, g, h in tree_edges:
        unique_nodes.add(parent)
        unique_nodes.add(child)
        node_data[child] = {"g": g, "h": h}

    node_levels = {root: 0}
    for parent, child, f, g, h in tree_edges:
        if parent in node_levels:
            node_levels[child] = node_levels[parent] + 1
        else:
            node_levels[parent] = 1
            node_levels[child] = 2

    level_counts = {}
    for node, lvl in node_levels.items():
        level_counts[lvl] = level_counts.get(lvl, 0) + 1

    max_nodes_in_any_level = max(level_counts.values()) if level_counts else 1
    level_current_idx = {}
    node_positions = {}

    for node, lvl in node_levels.items():
        idx = level_current_idx.get(lvl, 0)
        level_current_idx[lvl] = idx + 1
        total_in_lvl = level_counts[lvl]

        x_pos = (idx - (total_in_lvl - 1) / 2.0) * (15.0 / max(total_in_lvl, 1))
        if is_bfs and lvl > 1:
            x_pos = (idx - (total_in_lvl - 1) / 2.0) * 3.8

        y_pos = -lvl * 5.5
        node_positions[node] = (x_pos, y_pos)

    dynamic_width = max(8.5, max_nodes_in_any_level * 2.8 if is_bfs else 8.5)
    fig, ax = plt.subplots(figsize=(dynamic_width, 6.5))
    ax.set_title(title_name, fontsize=12, fontweight='bold', color='#2c3e50', pad=15)

    for parent, child, f, g, h in tree_edges:
        if parent in node_positions and child in node_positions:
            x1, y1 = node_positions[parent]
            x2, y2 = node_positions[child]
            ax.annotate("", xy=(x2, y2 + 1.2), xytext=(x1, y1 - 1.2),
                        arrowprops=dict(arrowstyle="-|>", color="#7f8c8d", lw=1.5, mutation_scale=12))

    for node, (x, y) in node_positions.items():
        box_w, box_h = 2.0, 2.0
        bg_color = '#e8f8f5' if node == root else ('#fdf2e9' if is_bfs else '#f4f6f7')
        edge_color = '#27ae60' if node == root else ('#d35400' if is_bfs else '#2980b9')

        rect = plt.Rectangle((x - box_w/2, y - box_h/2), box_w, box_h, facecolor=bg_color, edgecolor=edge_color, lw=2, zorder=3)
        ax.add_patch(rect)

        step_w = box_w / size
        step_h = box_h / size

        for i, val in enumerate(node):
            r, c = i // size, i % size
            tile_x = (x - box_w/2) + c * step_w
            tile_y = (y + box_h/2) - (r + 1) * step_h

            inner_rect = plt.Rectangle((tile_x, tile_y), step_w, step_h, facecolor='none', edgecolor='#bdc3c7', lw=0.6, zorder=4)
            ax.add_patch(inner_rect)

            txt_val = str(val) if val != 0 else ""
            if val == 0:
                blank_rect = plt.Rectangle((tile_x, tile_y), step_w, step_h, facecolor='#34495e', edgecolor='#bdc3c7', lw=0.6, zorder=4)
                ax.add_patch(blank_rect)

            ax.text(tile_x + step_w/2, tile_y + step_h/2, txt_val, ha='center', va='center', fontsize=8, fontweight='bold', color='#2c3e50' if val != 0 else 'white', zorder=5)

        info = node_data.get(node, {"g": 0, "h": 0})
        g_val, h_val = info["g"], info["h"]
        if is_bfs:
            lbl_txt = f"g(n) = {g_val}\n[Mù]"
        else:
            lbl_txt = f"g(n) = {g_val}, h(n) = {h_val:.1f}\nf(n) = {g_val + h_val:.1f}"

        ax.text(x, y - box_h/2 - 0.4, lbl_txt, ha='center', va='top', fontsize=8, color='#e67e22' if not is_bfs else '#d35400', fontweight='bold', zorder=6)

    all_x = [pos[0] for pos in node_positions.values()]
    all_y = [pos[1] for pos in node_positions.values()]
    if all_x and all_y:
        ax.set_xlim(min(all_x) - 2.5, max(all_x) + 2.5)
        ax.set_ylim(min(all_y) - 2.5, max(all_y) + 1)

    ax.axis('off')
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=130)
    plot_img = Image.open(buf)
    plt.close(fig)
    return plot_img


def draw_three_comparison_chart(a_nodes, a_time, ida_nodes, ida_time, bfs_nodes, bfs_time, bfs_timeout=False, ida_timeout=False):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 3.5), facecolor='#fdfefe')

    algorithms = ['A*', 'IDA*', 'BFS']
    node_counts = [a_nodes, ida_nodes, bfs_nodes]
    times = [a_time, ida_time, bfs_time]
    colors = ['#2ecc71', '#3498db', '#e74c3c']

    bars1 = ax1.bar(algorithms, node_counts, color=colors, width=0.45, edgecolor='#7f8c8d', lw=0.5)
    ax1.set_title('Số lượng Node cần duyệt', fontsize=10, fontweight='bold', color='#2c3e50')
    ax1.set_ylabel('Số lượng Node', fontsize=9)
    ax1.grid(axis='y', linestyle='--', alpha=0.5)

    for idx, bar in enumerate(bars1):
        yval = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2.0, yval + (yval * 0.02) + 1, f"{int(yval):,}", ha='center', va='bottom', fontweight='bold', fontsize=9)

    if bfs_timeout:
        ax1.text(2, bfs_nodes/2, "TIMEOUT\n(BFS)", ha='center', va='center', color='white', fontweight='bold', bbox=dict(facecolor='#c0392b', alpha=0.8, edgecolor='none'), fontsize=8)
    if ida_timeout:
        ax1.text(1, ida_nodes/2, "TIMEOUT\n(IDA*)", ha='center', va='center', color='white', fontweight='bold', bbox=dict(facecolor='#d35400', alpha=0.8, edgecolor='none'), fontsize=8)

    bars2 = ax2.bar(algorithms, times, color=colors, width=0.45, edgecolor='#7f8c8d', lw=0.5)
    ax2.set_title('Thời gian xử lý thuật toán (ms)', fontsize=10, fontweight='bold', color='#2c3e50')
    ax2.set_ylabel('Thời gian (ms)', fontsize=9)
    ax2.grid(axis='y', linestyle='--', alpha=0.5)

    for bar in bars2:
        yval = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2.0, yval + (yval * 0.02) + 0.1, f"{yval} ms", ha='center', va='bottom', fontweight='bold', fontsize=9)

    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=130)
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf)



# BƯỚC 5: ĐỒ HỌA & QUẢN LÝ GAME ENGINE


class DynamicAcademicPuzzleGame:
    def __init__(self):
        self.size = 3
        self.state = list(range(1, 8)) + [0, 8]
        self.last_shuffled_state = list(self.state)
        self.raw_image = None
        self.img_render_size = 450
        self.hint_idx = -1
        self.current_heuristic = "Manhattan"

        self.auto_play_steps = []
        self.current_play_index = 0
        self.is_playing = False

    def generate_board_image(self):
        tile_w = self.img_render_size // self.size
        if self.raw_image is None:
            base = Image.new("RGB", (self.img_render_size, self.img_render_size), color=(245, 245, 245))
            draw = ImageDraw.Draw(base)
            for i in range(self.size * self.size - 1):
                r, c = i // self.size, i % self.size
                draw.rectangle([c*tile_w, r*tile_w, (c+1)*tile_w, (r+1)*tile_w], fill=(41, 128, 185), outline=(255, 255, 255), width=2)
                draw.text((c*tile_w + tile_w//2 - 5, r*tile_w + tile_w//2 - 10), str(i+1), fill=(255, 255, 255))
            self.raw_image = base

        w, h = self.raw_image.size
        crop_side = min(w, h)
        img_square = self.raw_image.crop((0, 0, crop_side, crop_side)).resize((self.img_render_size, self.img_render_size))

        tiles = {}
        for idx in range(self.size * self.size - 1):
            r, c = idx // self.size, idx % self.size
            box = (c * tile_w, r * tile_w, (c + 1) * tile_w, (r + 1) * tile_w)
            tiles[idx + 1] = img_square.crop(box)

        tiles[0] = Image.new("RGB", (tile_w, tile_w), color=(44, 62, 80))

        board_img = Image.new("RGB", (self.img_render_size, self.img_render_size))
        for i, val in enumerate(self.state):
            r, c = i // self.size, i % self.size
            board_img.paste(tiles[val], (c * tile_w, r * tile_w))

        draw_grid = ImageDraw.Draw(board_img)
        for i in range(1, self.size):
            draw_grid.line([(i * tile_w, 0), (i * tile_w, self.img_render_size)], fill=(255, 255, 255), width=2)
            draw_grid.line([(0, i * tile_w), (self.img_render_size, i * tile_w)], fill=(255, 255, 255), width=2)

        if self.hint_idx != -1:
            hr, hc = self.hint_idx // self.size, self.hint_idx % self.size
            draw_grid.rectangle([hc * tile_w + 4, hr * tile_w + 4, (hc + 1) * tile_w - 4, (hr + 1) * tile_w - 4], outline=(46, 204, 113), width=6)
        return board_img

    def get_heuristic_status_log(self):
        misplaced = get_misplaced_tiles(self.state, self.size)
        manhattan = get_manhattan_distance(self.state, self.size)
        euclidean = get_euclidean_distance(self.state, self.size)
        conflict = get_linear_conflict(self.state, self.size)
        return f"THÔNG SỐ HÀM HEURISTIC TRÊN KHUNG MA TRẬN:\n- Ô sai vị trí (Misplaced Tiles): {misplaced}\n- Khoảng cách Manhattan: {manhattan}\n- Khoảng cách Euclid (Mới): {euclidean}\n- Xung đột chặn đầu (Linear Conflict): {conflict}\n-> Hàm chọn hiện tại [{self.current_heuristic}]: H(n) = {compute_heuristic(self.state, self.size, self.current_heuristic)}"

    def reset_graphs(self):
        return (draw_visual_search_tree([], "CÂY KHÔNG GIAN STATE-SPACE CỦA A*", self.size, self.current_heuristic),
                draw_visual_search_tree([], "CÂY KHÔNG GIAN DUYỆT MÙ CỦA BFS", self.size, is_bfs=True),
                draw_visual_search_tree([], "CÂY KHÔNG GIAN TÌM KIẾM CỦA IDA*", self.size, self.current_heuristic),
                draw_three_comparison_chart(0, 0, 0, 0, 0, 0))

    def upload_image(self, img):
        if img is not None:
            self.raw_image = img
        self.state = list(range(1, self.size * self.size)) + [0]
        self.last_shuffled_state = list(self.state)
        self.hint_idx = -1
        self.auto_play_steps = []
        self.is_playing = False
        t_a, t_b, t_c, chart = self.reset_graphs()
        return self.generate_board_image(), "Tải ảnh thành công! Bấm Xáo Trộn để chơi.", self.get_heuristic_status_log(), t_a, t_b, t_c, chart, gr.update(active=False)

    def change_grid(self, mode_str):
        self.size = 3 if mode_str == "3 × 3" else 4
        self.state = list(range(1, self.size * self.size)) + [0]
        self.last_shuffled_state = list(self.state)
        self.hint_idx = -1
        self.auto_play_steps = []
        self.is_playing = False
        t_a, t_b, t_c, chart = self.reset_graphs()
        return self.generate_board_image(), f"Đã chuyển cấu hình lưới: {mode_str}.", self.get_heuristic_status_log(), t_a, t_b, t_c, chart, gr.update(active=False)

    def change_heuristic_type(self, h_name):
        self.current_heuristic = h_name
        return self.get_heuristic_status_log()

    def shuffle_board(self):
        state = list(range(1, self.size * self.size)) + [0]
        shuffle_steps = 40 if self.size == 3 else 20

        for _ in range(shuffle_steps):
            blank = state.index(0)
            r, c = blank // self.size, blank % self.size
            moves = []
            if r > 0: moves.append(blank - self.size)
            if r < self.size - 1: moves.append(blank + self.size)
            if c > 0: moves.append(blank - 1)
            if c < self.size - 1: moves.append(blank + 1)
            chosen = random.choice(moves)
            state[blank], state[chosen] = state[chosen], state[blank]

        self.state = state
        self.last_shuffled_state = list(state)
        self.hint_idx = -1
        self.auto_play_steps = []
        self.is_playing = False
        t_a, t_b, t_c, chart = self.reset_graphs()
        return self.generate_board_image(), "Bàn cờ đã được xáo trộn mạnh mẽ tương ứng cấu hình lưới!", self.get_heuristic_status_log(), t_a, t_b, t_c, chart, gr.update(active=False)

    def back_to_last_shuffle(self):
        self.state = list(self.last_shuffled_state)
        self.hint_idx = -1
        self.auto_play_steps = []
        self.is_playing = False
        t_a, t_b, t_c, chart = self.reset_graphs()
        return self.generate_board_image(), "Đã quay trở lại trạng thái xáo trộn mới nhất!", self.get_heuristic_status_log(), t_a, t_b, t_c, chart, gr.update(active=False)

    def handle_mouse_click(self, evt: gr.SelectData):
        if self.is_playing:
            return gr.update(), "Đang chạy mô phỏng giải thuật, vui lòng đợi!", self.get_heuristic_status_log()
        click_x, click_y = evt.index[0], evt.index[1]
        tile_w = self.img_render_size // self.size
        col = click_x // tile_w
        row = click_y // tile_w
        clicked_idx = row * self.size + col
        blank_idx = self.state.index(0)

        if abs(row - (blank_idx // self.size)) + abs(col - (blank_idx % self.size)) == 1:
            self.state[blank_idx], self.state[clicked_idx] = self.state[clicked_idx], self.state[blank_idx]
            self.hint_idx = -1
            msg = "Đã dịch chuyển ô bằng chuột trái thành công."
        else:
            msg = "Hãy click vào những ô nằm sát cạnh ô trống."

        return self.generate_board_image(), msg, self.get_heuristic_status_log()

    def ai_hint_one_step(self):
        goal = list(range(1, self.size * self.size)) + [0]
        if self.state == goal:
            t_a, t_b, t_c, chart = self.reset_graphs()
            return self.generate_board_image(), "Bức tranh đã hoàn thành hoàn hảo!", self.get_heuristic_status_log(), t_a, t_b, t_c, chart, gr.update(active=False)

        path, a_nodes, a_duration, a_edges = solve_a_star_with_tree(self.state, self.size, self.current_heuristic)
        _, bfs_nodes, bfs_duration, bfs_edges = solve_bfs_with_tree(self.state, self.size, max_nodes=1500)
        _, ida_nodes, ida_duration, ida_edges = solve_ida_star_with_tree(self.state, self.size, self.current_heuristic, max_nodes=1500)

        tree_a_img = draw_visual_search_tree(a_edges, "CÂY KHÔNG GIAN STATE-SPACE CỦA A*", self.size, self.current_heuristic, is_bfs=False)
        tree_b_img = draw_visual_search_tree(bfs_edges, "CÂY KHÔNG GIAN DUYỆT MÙ CỦA BFS", self.size, self.current_heuristic, is_bfs=True)
        tree_c_img = draw_visual_search_tree(ida_edges, "CÂY KHÔNG GIAN TÌM KIẾM CỦA IDA*", self.size, self.current_heuristic, is_bfs=False)
        chart_img = draw_three_comparison_chart(a_nodes, a_duration, ida_nodes, ida_duration, bfs_nodes, bfs_duration)

        if not path or len(path) < 2:
            return self.generate_board_image(), "Không tìm thấy bước đi tiếp theo khả thi!", self.get_heuristic_status_log(), tree_a_img, tree_b_img, tree_c_img, chart_img, gr.update(active=False)

        next_state = path[1]
        tile_val = next_state[self.state.index(0)]
        self.hint_idx = self.state.index(tile_val)

        info = f"AI Gợi ý: Hãy bấm vào ô viền XANH LÁ (Mảnh số: {tile_val}). Đã cập nhật cây suy luận thời gian thực!"
        return self.generate_board_image(), info, self.get_heuristic_status_log(), tree_a_img, tree_b_img, tree_c_img, chart_img, gr.update(active=False)

    def ai_trigger_full_solve(self):
        goal = list(range(1, self.size * self.size)) + [0]
        if self.state == goal:
            t_a, t_b, t_c, chart = self.reset_graphs()
            return self.generate_board_image(), "Tranh đã xếp hoàn chỉnh rồi!", t_a, t_b, t_c, chart, "Hoàn thành!", gr.update(active=False)

        a_path, a_nodes, a_duration, a_edges = solve_a_star_with_tree(self.state, self.size, self.current_heuristic)
        bfs_path, bfs_nodes, bfs_duration, bfs_edges = solve_bfs_with_tree(self.state, self.size, max_nodes=30000)
        ida_path, ida_nodes, ida_duration, ida_edges = solve_ida_star_with_tree(self.state, self.size, self.current_heuristic, max_nodes=40000)

        bfs_timeout = (bfs_path is None)
        ida_timeout = (ida_path is None)

        bfs_msg = "TIMEOUT (>30,000 nodes)" if bfs_timeout else f"Duyệt {bfs_nodes} nodes | {bfs_duration} ms"
        ida_msg = "TIMEOUT (>40,000 nodes)" if ida_timeout else f"Duyệt {ida_nodes} nodes | {ida_duration} ms"

        if not a_path:
            t_a, t_b, t_c, chart = self.reset_graphs()
            return self.generate_board_image(), "Không tìm thấy đường đi giải!", t_a, t_b, t_c, chart, "Thất bại!", gr.update(active=False)

        self.auto_play_steps = a_path
        self.current_play_index = 0
        self.is_playing = True

        total_steps = len(a_path) - 1
        info = (f"ĐỐI SÁNH HIỆU NĂNG BA THUẬT TOÁN (Heuristic hiện tại: {self.current_heuristic}):\n"
                f"1. A* (Best-First): Duyệt {a_nodes} Nodes | Thời gian: {a_duration} ms (Đường đi: {total_steps} bước)\n"
                f"2. IDA* (Thử/Sai lặp): {ida_msg}\n"
                f"3. BFS (Duyệt mù): {bfs_msg}\n\n"
                f"Phân tích: Khoảng cách Euclid ước lượng dựa trên đường chéo hình học phẳng, nó có xu hướng cho ra giá trị nhỏ hơn Manhattan (Underestimate) nên an toàn nhưng số lượng Node mở rộng thường sẽ nhiều hơn Manhattan một chút.")

        tree_a_img = draw_visual_search_tree(a_edges, "CÂY SƠ ĐỒ TRẠNG THÁI CỦA A*", self.size, self.current_heuristic, is_bfs=False)
        tree_b_img = draw_visual_search_tree(bfs_edges, "CÂY KHÔNG GIAN TÌM MÙ CỦA BFS", self.size, self.current_heuristic, is_bfs=True)
        tree_c_img = draw_visual_search_tree(ida_edges, "CÂY KHÔNG GIAN TÌM KIẾM CỦA IDA*", self.size, self.current_heuristic, is_bfs=False)
        chart_img = draw_three_comparison_chart(a_nodes, a_duration, ida_nodes, ida_duration, bfs_nodes, bfs_duration, bfs_timeout, ida_timeout)

        return self.generate_board_image(), info, tree_a_img, tree_b_img, tree_c_img, chart_img, f"Bắt đầu bước 0/{total_steps}", gr.update(active=True)

    def auto_play_next_frame(self):
        if not self.is_playing or not self.auto_play_steps:
            return gr.update(), gr.update(), gr.update(), gr.update(active=False)

        if self.current_play_index >= len(self.auto_play_steps) - 1:
            self.is_playing = False
            return gr.update(), "Đã hoàn thành trình diễn chuỗi bước giải thuật Heuristic tối ưu trên bàn cờ!", "Hoàn thành!", gr.update(active=False)

        self.current_play_index += 1
        self.state = list(self.auto_play_steps[self.current_play_index])
        self.hint_idx = -1

        board_img = self.generate_board_image()
        status_text = f"ĐANG CHẠY BƯỚC GIẢI: Thực hiện bước {self.current_play_index} / {len(self.auto_play_steps)-1}..."
        progress_msg = f"Đang chạy bước {self.current_play_index}/{len(self.auto_play_steps)-1}"

        timer_update = gr.update(active=True) if self.current_play_index < len(self.auto_play_steps) - 1 else gr.update(active=False)
        return board_img, status_text, progress_msg, timer_update

game_engine = DynamicAcademicPuzzleGame()



# BƯỚC 6: XÂY DỰNG GIAO DIỆN GRADIO

custom_css = """
#btn_pink { background-color: #fce4ec !important; color: #880e4f !important; border: 1px solid #f8bbd0 !important; }
#btn_green { background-color: #e8f5e9 !important; color: #1b5e20 !important; border: 1px solid #c8e6c9 !important; }
"""

with gr.Blocks(theme=gr.themes.Soft(), css=custom_css) as demo:

    gr.Markdown("# 🧩DEMO TRÒ CHƠI GHÉP TRANH")

    auto_play_timer = gr.Timer(value=0.4, active=False)

    with gr.Row():
        with gr.Column(scale=4):
            game_board_display = gr.Image(
                value=game_engine.generate_board_image(),
                label="BÀN CỜ ĐỒ HỌA CHÍNH (Chọn tương tác ô kề cạnh)",
                type="pil", interactive=False, 
            )
            play_progress_bar = gr.Textbox(label="Tiến độ mô phỏng tự động giải bài toán", value="Chưa phát hành.", interactive=False)

            comparison_graph = gr.Image(
                value=draw_three_comparison_chart(0, 0, 0, 0, 0, 0),
                label="BIỂU ĐỒ ĐỐI SÁNH HIỆU NĂNG THỰC NGHIỆM ĐỊNH LƯỢNG ", type="pil", interactive=False
            )

        with gr.Column(scale=3):
            image_uploader = gr.Image(label="1. Tải hình ảnh tùy chỉnh", type="pil")
            grid_mode = gr.Radio(["3 × 3", "4 × 4"], label="2. Cấu hình kích thước ô lưới", value="3 × 3")


            heuristic_selector = gr.Dropdown(
                ["Manhattan", "Euclidean", "Linear Conflict", "Misplaced Tiles"],
                label="3. Chọn hàm Heuristic H(n) áp dụng",
                value="Manhattan"
            )

            with gr.Row():
                btn_shuffle = gr.Button("🎲 Xáo Trộn", elem_id="btn_pink")
                btn_back_shuffle = gr.Button("↩️ Quay lại", variant="secondary")
                btn_hint = gr.Button("💡 Gợi Ý 1 Bước", variant="primary")
                btn_solve_auto = gr.Button("🔥 AI Giải Toàn Bộ", elem_id="btn_green")

            action_log_box = gr.Textbox(label="Nhật ký ", value="Sẵn sàng. Hãy nhấn Xáo Trộn để bắt đầu.", lines=6)
            heuristic_stats_box = gr.Textbox(label="Phân tích tức thời hàm Heuristic H(n)", value=game_engine.get_heuristic_status_log(), lines=3, interactive=False)

    with gr.Row():
        with gr.Column(scale=4):
            visual_tree_graph_a = gr.Image(
                value=draw_visual_search_tree([], "CÂY KHÔNG GIAN STATE-SPACE CỦA A*", game_engine.size, game_engine.current_heuristic),
                label="CÂY SUY LUẬN TỐI ƯU CỦA A* ", type="pil", interactive=False
            )
        with gr.Column(scale=4):
            visual_tree_graph_c = gr.Image(
                value=draw_visual_search_tree([], "CÂY KHÔNG GIAN TÌM KIẾM CỦA IDA*", game_engine.size, game_engine.current_heuristic),
                label="CÂY SUY LUẬN CỦA IDA* ", type="pil", interactive=False
            )
        with gr.Column(scale=4):
            visual_tree_graph_b = gr.Image(
                value=draw_visual_search_tree([], "CÂY KHÔNG GIAN DUYỆT MÙ CỦA BFS", game_engine.size, is_bfs=True),
                label="CÂY DUYỆT MÙ TOÀN CỤC CỦA BFS ", type="pil", interactive=False
            )

    image_uploader.change(
        game_engine.upload_image, inputs=[image_uploader],
        outputs=[game_board_display, action_log_box, heuristic_stats_box, visual_tree_graph_a, visual_tree_graph_b, visual_tree_graph_c, comparison_graph, auto_play_timer]
    )
    grid_mode.change(
        game_engine.change_grid, inputs=[grid_mode],
        outputs=[game_board_display, action_log_box, heuristic_stats_box, visual_tree_graph_a, visual_tree_graph_b, visual_tree_graph_c, comparison_graph, auto_play_timer]
    )
    heuristic_selector.change(
        game_engine.change_heuristic_type, inputs=[heuristic_selector],
        outputs=[heuristic_stats_box]
    )
    btn_shuffle.click(
        game_engine.shuffle_board, inputs=[],
        outputs=[game_board_display, action_log_box, heuristic_stats_box, visual_tree_graph_a, visual_tree_graph_b, visual_tree_graph_c, comparison_graph, auto_play_timer]
    )
    btn_back_shuffle.click(
        game_engine.back_to_last_shuffle, inputs=[],
        outputs=[game_board_display, action_log_box, heuristic_stats_box, visual_tree_graph_a, visual_tree_graph_b, visual_tree_graph_c, comparison_graph, auto_play_timer]
    )
    btn_hint.click(
        game_engine.ai_hint_one_step, inputs=[],
        outputs=[game_board_display, action_log_box, heuristic_stats_box, visual_tree_graph_a, visual_tree_graph_b, visual_tree_graph_c, comparison_graph, auto_play_timer]
    )
    game_board_display.select(
        game_engine.handle_mouse_click, inputs=[],
        outputs=[game_board_display, action_log_box, heuristic_stats_box]
    )
    btn_solve_auto.click(
        game_engine.ai_trigger_full_solve, inputs=[],
        outputs=[game_board_display, action_log_box, visual_tree_graph_a, visual_tree_graph_b, visual_tree_graph_c, comparison_graph, play_progress_bar, auto_play_timer]
    )
    auto_play_timer.tick(
        game_engine.auto_play_next_frame, inputs=[],
        outputs=[game_board_display, action_log_box, play_progress_bar, auto_play_timer]
    )

demo.launch(inline=True, share=True)
